"""Shared plumbing for the Table-1 baselines (baselines_ml.py, baselines_gnn.py).

New file: nothing in tebnet/ or in scripts/train.py is modified. The TEB-Net package is only
IMPORTED (dataset loader, physics-feature names, DeepTherm split), so every baseline sees exactly the
same rows, units, splits and test molecules as TEB-Net.

A baseline run writes the same two files as scripts/train.py:
  <out_dir>/<tag>/<split_name>/member<m>_seed<s>/results.json
  <out_dir>/<tag>/<split_name>/member<m>_seed<s>/predictions.npz
so scripts/deeptherm_eval.py (including --combine and --exclude_file) and scripts/run_info.py read them
unchanged. Predictions are stored in SI units (J/mol, J/(mol·K)) like TEB-Net's; a target a run does not
predict is stored as NaN and simply shows up as NaN in the tables.
"""
from __future__ import annotations
import argparse
import atexit
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_ROOT = PROJECT_ROOT.parent
TEBNET_MODEL_ROOT = BENCHMARK_ROOT.parent.parent
for p in (PROJECT_ROOT, TEBNET_MODEL_ROOT, TEBNET_MODEL_ROOT / "data"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
KCAL_TO_J = 4184.0
CAL_TO_J = 4.184
J_TO_KCAL = 1.0 / KCAL_TO_J
J_TO_CAL = 1.0 / CAL_TO_J
DEEPTHERM_T_GRID = np.array([300.0, 400.0, 500.0, 600.0, 800.0, 1000.0, 1500.0])
DEEPTHERM_SI_ORDER_SHA1 = "102f81bdb66cc6fd825d519880d10b4f84523e7b"
TARGET_NAMES = ["H298"] + ["S298"] + [f"Cp{int(T)}" for T in DEEPTHERM_T_GRID]
GROUPS = {"h": [0], "s": [1], "cp": list(range(2, 9))}
def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--seed", type=int, default=0, help="model seed (initialisation / random search)")
    ap.add_argument("--csv", type=str, default=str(BENCHMARK_ROOT / "deeptherm_data.csv"))
    ap.add_argument("--physics_cache", type=str, default="auto")
    ap.add_argument("--smiles_col", type=str, default="smiles")
    ap.add_argument("--h_col", type=str, default=None)
    ap.add_argument("--s_col", type=str, default=None)
    ap.add_argument("--cp_cols", type=str, default=None)
    ap.add_argument("--h_unit", choices=["kcal", "kj", "j"], default=None)
    ap.add_argument("--s_unit", choices=["cal", "j"], default=None)
    ap.add_argument("--cp_unit", choices=["cal", "j"], default=None)
    ap.add_argument("--no_3d", action="store_true", help="drop the 3-D (moment-of-inertia) physics columns")
    ap.add_argument("--split_type", choices=["random", "complexity", "size", "scaffold", "functional_group"],
                    default="random")
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--split_fractions", type=float, nargs=3, default=[0.81, 0.09, 0.10])
    ap.add_argument("--test_smarts", type=str, default=None)
    ap.add_argument("--test_group", type=str, default=None, help="one of DeepTherm's functional-group names")
    ap.add_argument("--member", type=int, default=0, help="ensemble member (same meaning as in train.py)")
    ap.add_argument("--out_dir", type=str, default="benchmark_results_baselines")
    ap.add_argument("--tag_suffix", type=str, default="")
    ap.add_argument("--overwrite", action="store_true")
def order_fingerprint(h_j_per_mol: np.ndarray) -> str:
    s = ",".join(f"{round(float(v) * J_TO_KCAL, 2) + 0.0:.2f}" for v in h_j_per_mol)
    return hashlib.sha1(s.encode()).hexdigest()
def load_dataset(args) -> dict:
    """All rows of the CSV, in file order: SMILES, RDKit mols, targets in SI units and the V15 physics
    vector (from the physics cache). Row i here = row i of the CSV = index i in every split."""
    from tebnet.data.dataset import ThermochemDataset
    from tebnet.data.featurizer import V15_NAMES, FEATURES_VERSION
    cache = None if str(args.physics_cache).lower() == "none" else args.physics_cache
    base = ThermochemDataset.from_csv(
        args.csv, T_grid=DEEPTHERM_T_GRID, smiles_col=args.smiles_col, physics_cache=cache,
        h_col=args.h_col, s_col=args.s_col,
        cp_cols=[c.strip() for c in args.cp_cols.split(",")] if args.cp_cols else None,
        h_unit=args.h_unit, s_unit=args.s_unit, cp_unit=args.cp_unit,
        strict=True, compute_aux=False, compute_comp=False,
    )
    records = base.records
    full = ThermochemDataset(records, T_grid=DEEPTHERM_T_GRID, aux_feature_set="v15", compute_aux=True,
                             use_3d=not args.no_3d)
    if len(full.datapoints) != len(records):
        raise SystemExit(f"{len(records) - len(full.datapoints)} rows could not be parsed: row indices would no "
                         f"longer match the split. Fix the CSV first.")
    dps = full.datapoints
    H = np.array([d.dh_f_298 for d in dps], dtype=np.float64)
    data = {
        "smiles": [d.smiles for d in dps],
        "mols": [d.mol for d in dps],
        "H": H,
        "S": np.array([d.s_298 for d in dps], dtype=np.float64),
        "CP": np.stack([d.cp_grid for d in dps]).astype(np.float64),
        "X": np.stack([d.x_physic for d in dps]).astype(np.float64),
        "feature_names": list(V15_NAMES),
        "features_version": FEATURES_VERSION,
    }
    fp = order_fingerprint(H)
    data["order_sha1"] = fp
    data["same_order"] = fp == DEEPTHERM_SI_ORDER_SHA1
    if not data["same_order"]:
        print("WARNING: row order / values differ from DeepTherm Supplementary Data 1 — the split is still "
              "DeepTherm's algorithm, but not the same test molecules as the paper.")
    return data
def targets_si(data: dict) -> np.ndarray:
    """(n, 9) matrix in SI units: ΔH [J/mol], S [J/(mol K)], Cp(T) [J/(mol K)]."""
    return np.column_stack([data["H"], data["S"], data["CP"]])
def targets_dt(data: dict) -> np.ndarray:
    """(n, 9) matrix in DeepTherm units: ΔH [kcal/mol], S and Cp [cal/(mol K)]."""
    y = targets_si(data).copy()
    y[:, 0] *= J_TO_KCAL
    y[:, 1:] *= J_TO_CAL
    return y
def dt_to_si(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).copy()
    y[:, 0] *= KCAL_TO_J
    y[:, 1:] *= CAL_TO_J
    return y
def get_split(args, smiles: list[str]):
    from tebnet.data.splits import deeptherm_split, ensemble_member_split, FUNCTIONAL_GROUP_SMARTS
    test_smarts = args.test_smarts or (FUNCTIONAL_GROUP_SMARTS[args.test_group] if args.test_group else None)
    tr0, va0, te0 = deeptherm_split(smiles, args.split_type, tuple(args.split_fractions), args.split_seed,
                                    test_smarts)
    tr, va, te = ensemble_member_split(tr0, va0, te0, args.member, args.split_seed)
    split_name = args.split_type + (f"-{(args.test_group or test_smarts or '').replace(' ', '_')}"
                                    if args.split_type == "functional_group" else "") + f"{args.split_seed}"
    return np.asarray(tr), np.asarray(va), np.asarray(te), split_name, test_smarts
def prepare_run_dir(args, tag: str, split_name: str) -> Path:
    save_dir = PROJECT_ROOT / args.out_dir / tag / split_name / f"member{args.member}_seed{args.seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    if (save_dir / "results.json").exists() and not args.overwrite:
        print(f"skip: {save_dir} already has results.json (use --overwrite to redo it)")
        sys.exit(0)
    lock = save_dir / "RUNNING.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        sys.exit(f"another run is using {save_dir} (found {lock.name}). If none is active, delete that file.")
    atexit.register(lambda: lock.unlink(missing_ok=True))
    return save_dir
def _stats(pred, true, mask=None) -> dict:
    m = np.isfinite(true) & np.isfinite(pred)
    if mask is not None:
        m &= mask
    if not m.any():
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "n": 0}
    e = pred[m] - true[m]
    den = np.square(true[m] - true[m].mean()).sum()
    return {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt(np.square(e).mean())),
            "r2": float(1 - np.square(e).sum() / den) if den > 0 else float("nan"), "n": int(m.sum())}
def deeptherm_metrics(p: dict, t_grid: np.ndarray = DEEPTHERM_T_GRID, mask=None) -> dict:
    res = {"H298_kcal": _stats(p["h_pred"] * J_TO_KCAL, p["h_true"] * J_TO_KCAL, mask),
           "S298_cal": _stats(p["s_pred"] * J_TO_CAL, p["s_true"] * J_TO_CAL, mask)}
    for j, T in enumerate(t_grid):
        res[f"Cp{T:.0f}_cal"] = _stats(p["cp_pred"][:, j] * J_TO_CAL, p["cp_true"][:, j] * J_TO_CAL, mask)
    cp_maes = [res[f"Cp{T:.0f}_cal"]["mae"] for T in t_grid]
    with np.errstate(all="ignore"):
        res["Cp_mean_MAE_cal"] = float(np.nanmean(cp_maes)) if np.isfinite(cp_maes).any() else float("nan")
        vals = [res["H298_kcal"]["mae"], res["S298_cal"]["mae"], *cp_maes]
        res["mean9_MAE"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
    return res
def pred_dict(y_pred_si: np.ndarray, y_true_si: np.ndarray, x: np.ndarray, smiles: list[str]) -> dict:
    """Arrays in the layout deeptherm_eval.py expects."""
    return {"h_pred": y_pred_si[:, 0], "s_pred": y_pred_si[:, 1], "cp_pred": y_pred_si[:, 2:],
            "h_true": y_true_si[:, 0], "s_true": y_true_si[:, 1], "cp_true": y_true_si[:, 2:],
            "x": x.astype(np.float32), "smiles": list(smiles)}
def print_table(title: str, name: str, m_train: dict, m_val: dict, m_test: dict) -> None:
    print("\n" + "=" * 86)
    print(f" {title} | split {name}")
    print("=" * 86)
    print(f"  {'set':<8s} {'ΔH298':>8s} {'S298':>8s} {'Cp300':>8s} {'Cp mean':>8s} {'mean-9':>8s}")
    for lab, m in (("train", m_train), ("val", m_val), ("test", m_test)):
        print(f"  {lab:<8s} {m['H298_kcal']['mae']:>8.3f} {m['S298_cal']['mae']:>8.3f} "
              f"{m['Cp300_cal']['mae']:>8.3f} {m['Cp_mean_MAE_cal']:>8.3f} {m['mean9_MAE']:>8.3f}")
    print("=" * 86)
def write_run(save_dir: Path, *, args, tag: str, model: str, split_name: str, test_smarts, data: dict,
              idx: dict, preds: dict, extra: dict | None = None) -> dict:
    """preds = {"train": pred_dict, "val": ..., "test": ...}; idx = {"train": array, "val": ..., "test": ...}."""
    t_grid = DEEPTHERM_T_GRID
    m = {k: deeptherm_metrics(v, t_grid) for k, v in preds.items()}
    results = {
        "model": model, "tag": tag, "protocol": "deeptherm", "seed": args.seed, "member": args.member,
        "features_version": data["features_version"], "args": vars(args), "t_grid": t_grid.tolist(),
        "split": {"type": args.split_type, "seed": args.split_seed, "fractions": list(args.split_fractions),
                  "test_smarts": test_smarts, "same_order_as_deeptherm_SI": bool(data["same_order"]),
                  "order_sha1": data["order_sha1"]},
        "n_train": int(len(idx["train"])), "n_val": int(len(idx["val"])), "n_test": int(len(idx["test"])),
        "val_deeptherm_mae": m["val"]["mean9_MAE"],
        "train": m["train"], "val": m["val"], "test": m["test"],
        "test_indices": [int(i) for i in idx["test"]], "val_indices": [int(i) for i in idx["val"]],
        "train_indices": [int(i) for i in idx["train"]],
    }
    if extra:
        results.update(extra)
    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    pt, pv = preds["test"], preds["val"]
    np.savez(
        save_dir / "predictions.npz",
        **{f"test_{k}": v for k, v in pt.items() if k != "smiles"},
        **{f"val_{k}": v for k, v in pv.items() if k != "smiles"},
        test_smiles=np.array(pt["smiles"], dtype=object), val_smiles=np.array(pv["smiles"], dtype=object),
        test_indices=np.asarray(idx["test"]), t_grid=t_grid,
        feature_names=np.array(data["feature_names"], dtype=object),
    )
    print_table(f"{model} | {tag}", split_name, m["train"], m["val"], m["test"])
    print(f"  Results: {save_dir}/results.json\n")
    return results

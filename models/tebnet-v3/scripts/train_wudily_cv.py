"""TEB-Net on the WUDILY-CHO dataset under the k-fold protocol of Bruce et al.
("Machine learning predictions of thermochemical properties for aliphatic carbon and oxygen species",
Fuel 384 (2025) 133999; data: their Supplementary Material 1).

Protocol (Bruce et al., Section 2)
  * k-fold cross-validation: the dataset is split into k equal folds; each fold is the test set once and the
    other k - 1 folds are the training set.  Metrics are computed per fold and AVERAGED over the k folds.
  * metrics on the training set and on the test set: MAE, RMSE and R^2 ("Accuracy" in their Table 3),
    for ΔfH(298 K) in kcal/mol, S(298 K) in cal/(mol K) and Cp at 300-1500 K in cal/(mol K), Cp averaged over
    the temperatures (their Table 4).
  * their "chemical accuracy (95 % confidence interval)" equals 2 x test RMSE (2.21 = 2 x 1.10 kcal/mol for
    CDS-RF; 2.20 and 2.63 cal/(mol K) for S and Cp); printed here as "2 x RMSE".
  * k is NOT stated in the paper (it refers to Yalamanchi et al., J. Phys. Chem. A 2020); --k defaults to 5,
    the more conservative choice (80 % training data). Use the same k for every arm and report it.

What TEB-Net needs in addition (does not touch the test fold)
  * a validation subset for checkpoint selection: --val_frac (default 0.10) of the k - 1 training folds.
    Training metrics are still computed on all k - 1 folds, as in the paper.
  * folds are built over groups of identical molecules (canonical SMILES without stereo), so duplicates or
    stereoisomers never sit on both sides of a split; with a clean dataset this equals plain k-fold.
  * the fold assignment is saved once per dataset (folds_k<k>_cv<seed>.json) and reused by every arm.

Arms (as train_cv.py): scratch TEB-Net, --prior_only (frozen ridge prior alone), --init_ckpt (fine-tune).
Run the ΔH and S/Cp models separately and join them with --combine.

Typical use
  CSV=data/wudily_cho.csv
  for k in 0 1 2 3 4; do
    python scripts/train_wudily_cv.py --csv $CSV --folds $k --loss_weights 1 0 0   --monitor h   &
    python scripts/train_wudily_cv.py --csv $CSV --folds $k --loss_weights 0 1.5 1 --monitor scp &
  done; wait
  python scripts/train_wudily_cv.py --csv $CSV --prior_only
  python scripts/train_wudily_cv.py --csv $CSV --summarize
  python scripts/train_wudily_cv.py --csv $CSV --summarize --combine <H_TAG> <SCP_TAG>

Check the column mapping and the number of molecules printed at start-up (Bruce et al.: 6,074 species).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("tebnet_wudily")

J_TO_KCAL = 1.0 / 4184.0
J_TO_CAL = 1.0 / 4.184

# Published test/train metrics of Bruce et al. (2025), Tables 3 and 4, for the comparison printout.
# Note: in their Table 3 some rows have MAE > RMSE (e.g. MBTR-RF and SOAP-RF training columns), which is
# impossible, so only the two rows below are quoted.
PUBLISHED = {
    "CDS-RF (best)": {"H": dict(train_mae=0.82, test_mae=0.84, train_rmse=1.08, test_rmse=1.10, train_r2=0.99),
                      "S": dict(train_mae=0.83, test_mae=0.83, train_rmse=1.09, test_rmse=1.10, train_r2=0.99, test_r2=0.99),
                      "Cp": dict(train_mae=0.36, test_mae=0.86, train_rmse=0.55, test_rmse=1.31, train_r2=0.99, test_r2=0.99)},
    "GNF-MPNN (best graph network)": {"H": dict(train_mae=3.58, test_mae=6.45, train_rmse=5.21, test_rmse=10.19,
                                                train_r2=0.99)},
}


# ───────────────────────────── folds ─────────────────────────────
def group_keys(smiles: list[str]) -> list[str]:
    """Canonical SMILES without stereo (radicals kept); falls back to the input string."""
    try:
        from rdkit import Chem
    except ImportError:
        return [s.strip() for s in smiles]
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        out.append(Chem.MolToSmiles(m, isomericSmiles=False) if m is not None else s.strip())
    return out


def grouped_kfold(keys: list[str], k: int, seed: int) -> list[np.ndarray]:
    """k folds of near-equal size; molecules with the same key always share a fold."""
    groups: dict[str, list[int]] = {}
    for i, g in enumerate(keys):
        groups.setdefault(g, []).append(i)
    order = list(groups.values())
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    order.sort(key=len, reverse=True)          # place multi-member groups first, then fill evenly
    folds: list[list[int]] = [[] for _ in range(k)]
    for members in order:
        j = min(range(k), key=lambda f: (len(folds[f]), f))
        folds[j].extend(members)
    return [np.array(sorted(f), dtype=int) for f in folds]


def load_or_make_folds(path: Path, smiles: list[str], k: int, seed: int) -> list[np.ndarray]:
    if path.exists():
        d = json.loads(path.read_text())
        if d["smiles"] != smiles:
            sys.exit(f"{path} was made for a different molecule list; delete it or use another --cv_seed")
        return [np.asarray(f, dtype=int) for f in d["folds"]]
    keys = group_keys(smiles)
    folds = grouped_kfold(keys, k, seed)
    n_dup = len(keys) - len(set(keys))
    path.write_text(json.dumps({"k": k, "cv_seed": seed, "n": len(smiles), "n_duplicate_members": n_dup,
                                "smiles": smiles, "folds": [f.tolist() for f in folds]}))
    logger.info(f"wrote {path}: {k} folds, sizes {[len(f) for f in folds]}, "
                f"{n_dup} molecules share a canonical SMILES with another one (kept in the same fold)")
    return folds


def val_split(pool: np.ndarray, keys: list[str], val_frac: float, member: int, seed: int, fold: int) -> np.ndarray:
    """Validation subset of the training pool for member `member` (grouped, disjoint across members)."""
    n_parts = max(2, int(round(1.0 / val_frac)))
    parts = grouped_kfold([keys[i] for i in pool], n_parts, 1000 * (seed + 1) + fold)
    return np.sort(pool[parts[member % n_parts]])


# ───────────────────────────── metrics (pure numpy) ─────────────────────────────
def mae_rmse_r2(pred, true):
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    m = np.isfinite(pred) & np.isfinite(true)
    if not m.any():
        return float("nan"), float("nan"), float("nan")
    e = pred[m] - true[m]
    ss_tot = np.square(true[m] - true[m].mean()).sum()
    r2 = 1.0 - np.square(e).sum() / ss_tot if ss_tot > 0 else float("nan")
    return float(np.abs(e).mean()), float(np.sqrt(np.square(e).mean())), float(r2)


def bruce_metrics(p: dict, t_grid: np.ndarray) -> dict:
    """p in SI units. Returns {prop: (MAE, RMSE, R2)} in kcal/mol and cal/(mol K); Cp = mean over T."""
    out = {"H": mae_rmse_r2(p["h_pred"] * J_TO_KCAL, p["h_true"] * J_TO_KCAL),
           "S": mae_rmse_r2(p["s_pred"] * J_TO_CAL, p["s_true"] * J_TO_CAL)}
    per_t = [mae_rmse_r2(p["cp_pred"][:, j] * J_TO_CAL, p["cp_true"][:, j] * J_TO_CAL) for j in range(len(t_grid))]
    out["Cp"] = tuple(float(np.nanmean([x[i] for x in per_t])) for i in range(3))
    for j, T in enumerate(t_grid):
        out[f"Cp{T:.0f}"] = per_t[j]
    return out


def fold_table(per_fold: list[dict], title: str, k: int) -> None:
    """Mean ± s.d. over folds, the way Bruce et al. average their k iterations."""
    def ms(split, prop, i):
        v = np.array([f[split][prop][i] for f in per_fold], float)
        return f"{np.nanmean(v):.2f} ± {np.nanstd(v):.2f}" if len(v) > 1 else f"{v[0]:.2f}"

    print(f"\n  {title}  ({len(per_fold)}/{k} folds; mean ± s.d. over folds)")
    print("  | Property | Train MAE | Test MAE | Train RMSE | Test RMSE | Train R² | Test R² | 2 × test RMSE |")
    print("  |---|---|---|---|---|---|---|---|")
    for prop, label in (("H", "ΔfH°298 (kcal/mol)"), ("S", "S°298 (cal/(mol K))"), ("Cp", "Cp 300–1500 K mean (cal/(mol K))")):
        rmse = np.nanmean([f["test"][prop][1] for f in per_fold])
        print(f"  | {label} | {ms('train', prop, 0)} | {ms('test', prop, 0)} | {ms('train', prop, 1)} | "
              f"{ms('test', prop, 1)} | {ms('train', prop, 2)} | {ms('test', prop, 2)} | {2 * rmse:.2f} |")


def published_table() -> None:
    print("\n  Published, Bruce et al. (2025) Tables 3–4 (same units; test = averaged over their k folds)")
    print("  | Model | Property | Train MAE | Test MAE | Train RMSE | Test RMSE | Train R² | Test R² | 2 × test RMSE |")
    print("  |---|---|---|---|---|---|---|---|---|")
    for model, props in PUBLISHED.items():
        for prop, d in props.items():
            g = lambda key: f"{d[key]:.2f}" if key in d else "–"
            print(f"  | {model} | {prop} | {g('train_mae')} | {g('test_mae')} | {g('train_rmse')} | {g('test_rmse')} | "
                  f"{g('train_r2')} | {g('test_r2')} | {2 * d['test_rmse']:.2f} |")


# ───────────────────────────── structural classes (for the error breakdown) ─────────────────────────────
CLASS_ORDER = ["all", "radical", "closed shell", "contains O", "C/H only", "acyclic", "one ring", "two or more rings",
               "aromatic", "fused aromatic (PAH type)", "two or more non-fused benzene rings", "ring of 8-11 atoms",
               "macrocycle (ring of ≥12 atoms)", "epoxide", "ether (non-epoxide)", "1,2-diol", "vinylogous acid",
               "quaternary carbon (highly branched)", "1-5 heavy atoms", "6-10 heavy atoms", "11-14 heavy atoms",
               "15-19 heavy atoms", "≥20 heavy atoms", "0-2 rotatable bonds", "3-5 rotatable bonds",
               "6 or more rotatable bonds"]


def structural_classes(smiles: str) -> set[str]:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors as D
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return {"all"}
    c = {"all"}
    c.add("radical" if any(a.GetNumRadicalElectrons() for a in m.GetAtoms()) else "closed shell")
    c.add("contains O" if any(a.GetSymbol() == "O" for a in m.GetAtoms()) else "C/H only")
    rings = m.GetRingInfo().AtomRings()
    c.add("acyclic" if not rings else "one ring" if len(rings) == 1 else "two or more rings")
    arom = [set(r) for r in rings if all(m.GetAtomWithIdx(i).GetIsAromatic() for i in r)]
    if arom:
        c.add("aromatic")
    if any(len(a & b) >= 2 for i, a in enumerate(arom) for b in arom[i + 1:]):
        c.add("fused aromatic (PAH type)")
    benz = [r for r in arom if len(r) == 6 and all(m.GetAtomWithIdx(i).GetSymbol() == "C" for i in r)]
    if len(benz) >= 2 and not any(a & b for i, a in enumerate(benz) for b in benz[i + 1:]):
        c.add("two or more non-fused benzene rings")
    if any(8 <= len(r) <= 11 for r in rings):
        c.add("ring of 8-11 atoms")
    if any(len(r) >= 12 for r in rings):
        c.add("macrocycle (ring of ≥12 atoms)")
    patt = {"epoxide": "C1OC1", "ether (non-epoxide)": "[OX2;!r3]([#6;!$(C=O)])[#6;!$(C=O)]",
            "1,2-diol": "[OX2H][CX4][CX4][OX2H]", "vinylogous acid": "[OX2H][CX3]=[CX3][CX3]=[OX1]",
            "quaternary carbon (highly branched)": "[CX4H0]([#6])([#6])([#6])[#6]"}
    for name, sm in patt.items():
        q = Chem.MolFromSmarts(sm)
        if q is not None and m.HasSubstructMatch(q):
            c.add(name)
    n = m.GetNumHeavyAtoms()
    c.add("1-5 heavy atoms" if n <= 5 else "6-10 heavy atoms" if n <= 10 else "11-14 heavy atoms" if n <= 14
          else "15-19 heavy atoms" if n <= 19 else "≥20 heavy atoms")
    r = D.CalcNumRotatableBonds(m)
    c.add("0-2 rotatable bonds" if r <= 2 else "3-5 rotatable bonds" if r <= 5 else "6 or more rotatable bonds")
    return c


def class_table(smiles: list[str], p: dict, title: str) -> None:
    """Test MAE per structural class, every molecule counted once (pooled over the folds)."""
    try:
        cls = [structural_classes(s) for s in smiles]
    except ImportError:
        print("\n  (RDKit not available: structural breakdown skipped)")
        return
    eh = np.abs(p["h_pred"] - p["h_true"]) * J_TO_KCAL
    es = np.abs(p["s_pred"] - p["s_true"]) * J_TO_CAL
    ec = np.abs(p["cp_pred"] - p["cp_true"]).mean(1) * J_TO_CAL
    print(f"\n  {title}: test MAE by structural class (each molecule once, from the fold where it was tested)")
    print("  | Class | n | ΔfH°298 (kcal/mol) | S°298 (cal/(mol K)) | Cp mean (cal/(mol K)) |")
    print("  |---|---|---|---|---|")
    for name in CLASS_ORDER:
        m = np.array([name in c for c in cls])
        if m.sum() == 0:
            continue
        print(f"  | {name} | {int(m.sum())} | {np.nanmean(eh[m]):.2f} | {np.nanmean(es[m]):.2f} | {np.nanmean(ec[m]):.2f} |")


def worst_table(smiles: list[str], p: dict, n: int = 15) -> None:
    e = (p["h_pred"] - p["h_true"]) * J_TO_KCAL
    order = np.argsort(-np.abs(e))[:n]
    print(f"\n  Largest ΔfH°298 errors (kcal/mol)")
    print("  | SMILES | reference | TEB-Net | error |")
    print("  |---|---|---|---|")
    for i in order:
        print(f"  | {smiles[i]} | {p['h_true'][i] * J_TO_KCAL:.2f} | {p['h_pred'][i] * J_TO_KCAL:.2f} | {e[i]:+.2f} |")


# ───────────────────────────── loading results ─────────────────────────────
def load_tag(run_dir: Path):
    files = sorted(run_dir.glob("fold*.npz"))
    if not files:
        return None
    zs = [np.load(f, allow_pickle=True) for f in files]
    meta = json.loads(str(zs[0]["meta"]))
    per_fold = [json.loads(str(z["metrics"])) for z in zs]
    idx = np.concatenate([z["test_idx"] for z in zs])
    o = np.argsort(idx)
    pooled = {key: np.concatenate([z[key] for z in zs])[o]
              for key in ("h_pred", "s_pred", "cp_pred", "h_true", "s_true", "cp_true")}
    smiles = list(np.concatenate([z["smiles"] for z in zs])[o])
    folds = {int(json.loads(str(z["meta"]))["fold"]): z for z in zs}
    return dict(meta=meta, per_fold=per_fold, pooled=pooled, smiles=smiles, idx=idx[o], folds=folds,
                n_files=len(files))


def summarize(args, root: Path) -> None:
    tags = sorted(d for d in root.iterdir() if d.is_dir()) if root.exists() else []
    loaded = {}
    for d in tags:
        r = load_tag(d)
        if r is None:
            continue
        loaded[d.name] = r
        t_grid = np.asarray(r["meta"]["t_grid"])
        print("\n" + "=" * 100 + f"\n {d.name}\n" + "=" * 100)
        fold_table(r["per_fold"], f"{r['meta']['mode']} | weights {r['meta']['loss_weights']}", r["meta"]["k"])
        pooled = bruce_metrics(r["pooled"], t_grid)
        print("  pooled test (every molecule once): " + " | ".join(
            f"{k} MAE {pooled[k][0]:.2f}, RMSE {pooled[k][1]:.2f}" for k in ("H", "S", "Cp")))
    if not loaded:
        sys.exit(f"no runs under {root}")
    if args.combine:
        th, ts = args.combine
        if th not in loaded or ts not in loaded:
            sys.exit(f"--combine: tag not found. Available: {sorted(loaded)}")
        rh, rs = loaded[th], loaded[ts]
        if not np.array_equal(rh["idx"], rs["idx"]) or set(rh["folds"]) != set(rs["folds"]):
            sys.exit("--combine: the two tags do not cover the same folds")
        t_grid = np.asarray(rs["meta"]["t_grid"])
        per_fold = []
        for f in sorted(rh["folds"]):
            zh, zs = rh["folds"][f], rs["folds"][f]
            if not (np.array_equal(zh["test_idx"], zs["test_idx"]) and np.array_equal(zh["train_idx"], zs["train_idx"])):
                sys.exit(f"--combine: fold {f} differs between the tags")
            m = {}
            for split, pre in (("test", ""), ("train", "train_")):
                p = {key: zs[pre + key] for key in ("s_pred", "cp_pred", "h_true", "s_true", "cp_true")}
                p["h_pred"] = zh[pre + "h_pred"]
                m[split] = bruce_metrics(p, t_grid)
            per_fold.append(m)
        p = dict(rs["pooled"])
        p["h_pred"] = rh["pooled"]["h_pred"]
        print("\n" + "#" * 100 + f"\n DECOUPLED TEB-Net   ΔH <- {th}\n                     S, Cp <- {ts}\n" + "#" * 100)
        fold_table(per_fold, "decoupled TEB-Net, Bruce et al. protocol", rs["meta"]["k"])
        published_table()
        pooled = bruce_metrics(p, t_grid)
        print("\n  Cp test MAE by temperature (pooled, cal/(mol K)): " + "  ".join(
            f"{T:.0f} K {pooled[f'Cp{T:.0f}'][0]:.2f}" for T in t_grid))
        class_table(rs["smiles"], p, "decoupled TEB-Net")
        if "prior" in {r["meta"]["mode"] for r in loaded.values()}:
            for name, r in loaded.items():
                if r["meta"]["mode"] == "prior" and np.array_equal(r["idx"], rs["idx"]):
                    class_table(r["smiles"], r["pooled"], f"prior only ({name})")
        worst_table(rs["smiles"], p)
        out = root / f"predictions_{th}__{ts}.csv"
        import pandas as pd
        df = pd.DataFrame({"smiles": rs["smiles"],
                           "H_ref_kcal": p["h_true"] * J_TO_KCAL, "H_teb_kcal": p["h_pred"] * J_TO_KCAL,
                           "S_ref_cal": p["s_true"] * J_TO_CAL, "S_teb_cal": p["s_pred"] * J_TO_CAL})
        for j, T in enumerate(t_grid):
            df[f"Cp{T:.0f}_ref_cal"] = p["cp_true"][:, j] * J_TO_CAL
            df[f"Cp{T:.0f}_teb_cal"] = p["cp_pred"][:, j] * J_TO_CAL
        df.to_csv(out, index=False, float_format="%.4f")
        print(f"\n  per-molecule test predictions: {out}")


# ───────────────────────────── main ─────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="TEB-Net, k-fold protocol of Bruce et al. (2025) on WUDILY-CHO")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--smiles_col", default="smiles")
    ap.add_argument("--h_col", default=None)
    ap.add_argument("--s_col", default=None)
    ap.add_argument("--cp_cols", default=None, help="comma-separated, one per T in 300,400,500,600,800,1000,1500 K")
    ap.add_argument("--h_unit", choices=["kcal", "kj", "j"], default="kcal")
    ap.add_argument("--s_unit", choices=["cal", "j"], default="cal")
    ap.add_argument("--cp_unit", choices=["cal", "j"], default="cal")
    ap.add_argument("--physics_cache", default="auto")
    ap.add_argument("--strict", action="store_true", help="stop on a molecule the featurizer rejects "
                                                          "(default: drop it and list it)")
    ap.add_argument("--out_dir", default="benchmark_results_wudily")
    ap.add_argument("--tag_suffix", default="")
    # protocol
    ap.add_argument("--k", type=int, default=5, help="number of folds (not stated by Bruce et al.; see docstring)")
    ap.add_argument("--folds", type=int, nargs="*", default=None, help="subset of folds to run")
    ap.add_argument("--cv_seed", type=int, default=0, help="fold assignment (keep fixed across arms)")
    ap.add_argument("--val_frac", type=float, default=0.10, help="validation share of the training folds")
    ap.add_argument("--n_members", type=int, default=1, help="ensemble members per fold (different val subsets)")
    ap.add_argument("--seed", type=int, default=0, help="model initialisation")
    # arms
    ap.add_argument("--prior_only", action="store_true")
    ap.add_argument("--init_ckpt", default=None)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--combine", nargs=2, metavar=("H_TAG", "SCP_TAG"), default=None)
    # model / training: the DeepTherm settings of the paper (batch 32, 100 epochs; ~4,900 training molecules)
    ap.add_argument("--loss_weights", type=float, nargs=3, default=[2.0, 1.5, 1.0], metavar=("H", "S", "CP"))
    ap.add_argument("--monitor", choices=["deeptherm", "h", "s", "cp300", "scp"], default="deeptherm")
    ap.add_argument("--atom_featurizer", choices=["chemprop", "radical", "thermo"], default="thermo")
    ap.add_argument("--baseline", choices=["none", "composition", "physics"], default="physics")
    ap.add_argument("--no_3d", action="store_true")
    ap.add_argument("--route_mode", choices=["grouped", "full"], default="grouped")
    ap.add_argument("--ridge_lambda", type=float, default=1e-3)
    ap.add_argument("--loss_norm", choices=["global", "residual"], default="global")
    ap.add_argument("--mp_depth", type=int, default=5)
    ap.add_argument("--mp_hidden", type=int, default=300)
    ap.add_argument("--branch_hidden", type=int, default=256)
    ap.add_argument("--branch_layers", type=int, default=2)
    ap.add_argument("--branch_dropout", type=float, default=0.1)
    ap.add_argument("--t_mid", type=float, default=1000.0)
    ap.add_argument("--t_high_dom", type=float, default=3000.0)
    ap.add_argument("--max_epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--warmup_epochs", type=int, default=3)
    ap.add_argument("--init_lr", type=float, default=1e-4)
    ap.add_argument("--max_lr", type=float, default=8e-4)
    ap.add_argument("--final_lr", type=float, default=1e-5)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--keep_ckpt", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    # v3 upgrades (all off by default = v1 behaviour)
    ap.add_argument("--prior_groups", action="store_true",
                    help="v3: add Benson-type group counts and ring-correction keys to the frozen prior")
    ap.add_argument("--group_min_count", type=int, default=3,
                    help="v3: groups seen in fewer training molecules get no prior coefficient")
    ap.add_argument("--huber_delta_h", type=float, default=None, help="v3: Huber delta for dfH, kcal/mol")
    ap.add_argument("--huber_delta_s", type=float, default=None, help="v3: Huber delta for S, cal/(mol K)")
    ap.add_argument("--huber_delta_cp", type=float, default=None, help="v3: Huber delta for Cp, cal/(mol K)")
    from tebnet.data.featurizer import S_PRIOR_VARIANTS, ENTROPY_V2_NAMES
    ap.add_argument("--s_prior", choices=S_PRIOR_VARIANTS, default="v1",
                    help="Entropy prior variant (v2_exact fixes S_trans, S_rot, sigma, spin)")
    args = ap.parse_args()

    ds_name = Path(args.csv).stem
    root = Path(args.out_dir) / ds_name
    if args.summarize:
        summarize(args, root)
        return

    # heavy imports only for training (the summary needs numpy, pandas and RDKit only)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    import torch
    from torch.utils.data import DataLoader, Subset
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_cv import arrays, fit_priors, prior_predict, train_inner   # also sets the project paths
    from train import make_atom_featurizer, collect_predictions, COL, DEEPTHERM_T_GRID
    from tebnet.data.dataset import ThermochemDataset
    from tebnet.data.collate import ThermoCollater
    from tebnet.data.featurizer import FEATURES_VERSION

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_3d = not args.no_3d
    t_grid = DEEPTHERM_T_GRID
    t_mid = args.t_mid
    t_low_dom = float(t_grid[t_grid <= t_mid].min())

    mode = "prior" if args.prior_only else ("finetune" if args.init_ckpt else "scratch")
    tag = mode
    if mode != "prior":
        tag += ("" if args.loss_weights == [2.0, 1.5, 1.0] else "_w" + "-".join(f"{w:g}" for w in args.loss_weights))
        tag += ("" if args.monitor == "deeptherm" else f"_mon-{args.monitor}")
        tag += f"_{args.atom_featurizer}_d{args.mp_depth}_ep{args.max_epochs}_m{args.n_members}"
    tag += f"_{args.baseline}" + ("" if use_3d else "_2d") + f"_k{args.k}_cv{args.cv_seed}_s{args.seed}"
    if args.prior_groups:
        tag += f"_grp{args.group_min_count}"
    if mode != "prior" and any(d is not None for d in (args.huber_delta_h, args.huber_delta_s, args.huber_delta_cp)):
        tag += "_hub" + "-".join("x" if d is None else f"{d:g}"
                                 for d in (args.huber_delta_h, args.huber_delta_s, args.huber_delta_cp))
    tag += args.tag_suffix
    run_dir = root / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    full_ds = ThermochemDataset.from_csv(
        args.csv, T_grid=t_grid, smiles_col=args.smiles_col, physics_cache=args.physics_cache,
        h_col=args.h_col, s_col=args.s_col,
        cp_cols=[c.strip() for c in args.cp_cols.split(",")] if args.cp_cols else None,
        h_unit=args.h_unit, s_unit=args.s_unit, cp_unit=args.cp_unit, strict=args.strict,
        compute_aux=True, aux_feature_set="v15", use_3d=use_3d, compute_comp=False,
    )
    dps = full_ds.datapoints
    smiles = [dp.smiles for dp in dps]
    import pandas as pd
    raw = pd.read_csv(args.csv)[args.smiles_col].astype(str).tolist()
    dropped = sorted(set(raw) - set(smiles)) if len(raw) != len(smiles) else []
    logger.info(f"{ds_name}: {len(raw)} rows in the CSV, {len(smiles)} featurised"
                + (f", {len(dropped)} dropped (listed in {root / 'dropped.txt'})" if dropped else ""))
    if dropped:
        (root / "dropped.txt").write_text("\n".join(dropped) + "\n")

    folds = load_or_make_folds(root / f"folds_k{args.k}_cv{args.cv_seed}.json", smiles, args.k, args.cv_seed)
    if args.prior_groups:
        # v3: append the group / ring-correction counts to every descriptor vector; the network branches keep
        # their v15 inputs, only the frozen prior gets the new columns (see train_cv.fit_priors)
        from tebnet.data.groups import load_or_compute_groups
        G, vocab = load_or_compute_groups(args.csv, smiles)
        for dp, g in zip(dps, G):
            dp.x_physic = np.concatenate([dp.x_physic[:len(COL)], g]).astype(np.float32)
        args.feature_names = list(COL) + vocab
        args.extra_base_names = vocab
        logger.info(f"v3 prior: +{len(vocab)} group / ring-correction columns "
                    f"(min training count {args.group_min_count})")
    if getattr(args, "s_prior", "v1") != "v1":
        from tebnet.data.featurizer import load_or_compute_entropy_v2, entropy_v2_cache_path
        cache_path = entropy_v2_cache_path(args.csv)
        ev2 = load_or_compute_entropy_v2(smiles, cache_path=cache_path, n_conf=10, seed=args.seed, workers=args.num_workers)
        for dp, g in zip(dps, ev2):
            dp.x_physic = np.concatenate([dp.x_physic, g]).astype(np.float32)
        args.feature_names = list(getattr(args, "feature_names", COL)) + ENTROPY_V2_NAMES
        logger.info(f"S prior {args.s_prior}: +{len(ENTROPY_V2_NAMES)} physical entropy columns appended")
    keys = group_keys(smiles)
    atom_feat = make_atom_featurizer(args.atom_featurizer)
    collater = ThermoCollater(atom_featurizer=atom_feat)
    d_v = len(atom_feat) if atom_feat is not None else None
    if use_3d and max(dp.x_physic[COL["moi_valid"]] for dp in dps) < 0.5:
        logger.warning("use_3d=True but no valid 3-D features")

    for k in (args.folds if args.folds else range(args.k)):
        out_file = run_dir / f"fold{k}.npz"
        if out_file.exists() and not args.overwrite:
            logger.info(f"fold {k} already done ({out_file}); skipping")
            continue
        te = folds[k]
        tr_all = np.sort(np.concatenate([f for j, f in enumerate(folds) if j != k]))
        t0 = time.time()
        if args.prior_only:
            X, H, S, CP = arrays(dps, tr_all)
            _, cols, bh, bs, bc = fit_priors(args, X, H, S, CP, t_grid, t_mid, t_low_dom, use_3d)
            pred = {}
            for name, idx in (("test", te), ("train", tr_all)):
                h, s, cp = prior_predict(arrays(dps, idx)[0], cols, bh, bs, bc, t_grid, t_mid, t_low_dom,
                                         args.t_high_dom)
                pred[name] = {"h_pred": h, "s_pred": s, "cp_pred": cp}
        else:
            loader = lambda idx: DataLoader(Subset(full_ds, idx), batch_size=args.batch_size, shuffle=False,
                                            collate_fn=collater, num_workers=args.num_workers)
            acc = {"test": [], "train": []}
            for mbr in range(args.n_members):
                va = val_split(tr_all, keys, args.val_frac, mbr, args.cv_seed, k)
                tr = np.setdiff1d(tr_all, va)
                save_dir = run_dir / f"fold{k}" / f"member{mbr}"
                save_dir.mkdir(parents=True, exist_ok=True)
                model, best_val, *_ = train_inner(args, full_ds, tr, va, args.seed + 100 * k + mbr, str(save_dir),
                                                  t_grid, t_mid, t_low_dom, use_3d, collater, d_v, device)
                for name, idx in (("test", te), ("train", tr_all)):
                    acc[name].append(collect_predictions(model, loader(idx), device))
                logger.info(f"fold {k} member {mbr}: {len(tr)} train / {len(va)} val / {len(te)} test | "
                            f"best val {best_val:.3f}")
                if not args.keep_ckpt:
                    shutil.rmtree(save_dir, ignore_errors=True)
                del model
                torch.cuda.empty_cache()
            pred = {name: {key: np.mean([a[key] for a in acc[name]], axis=0) for key in ("h_pred", "s_pred", "cp_pred")}
                    for name in acc}
        truth = {}
        for name, idx in (("test", te), ("train", tr_all)):
            _, H, S, CP = arrays(dps, idx)
            truth[name] = {"h_true": H, "s_true": S, "cp_true": CP}
        metrics = {name: bruce_metrics({**pred[name], **truth[name]}, t_grid) for name in ("train", "test")}
        meta = {"mode": mode, "tag": tag, "k": args.k, "fold": k, "cv_seed": args.cv_seed, "val_frac": args.val_frac,
                "n_members": args.n_members, "loss_weights": args.loss_weights, "monitor": args.monitor,
                "t_grid": t_grid.tolist(), "features_version": FEATURES_VERSION, "args": vars(args)}
        np.savez(out_file, test_idx=te, train_idx=tr_all, smiles=np.array([smiles[i] for i in te], dtype=object),
                 **pred["test"], **truth["test"],
                 **{"train_" + key: v for key, v in {**pred["train"], **truth["train"]}.items()},
                 metrics=json.dumps(metrics), meta=json.dumps(meta))
        fold_table([metrics], f"fold {k}: {len(tr_all)} train / {len(te)} test | {time.time() - t0:.0f} s", args.k)

    if all((run_dir / f"fold{k}.npz").exists() for k in range(args.k)):
        r = load_tag(run_dir)
        print("\n" + "=" * 100 + f"\n {ds_name} | {tag} | all {args.k} folds\n" + "=" * 100)
        fold_table(r["per_fold"], "Bruce et al. protocol", args.k)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()
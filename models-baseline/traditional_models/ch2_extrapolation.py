"""CH2 extrapolation benchmark: every Table-1 baseline scored on the five homologous series.

Rebuilds the exact models of ``benchmark_results_baselines`` for the DeepTherm random split 0
(conventional ML: refit from the ``best_params`` recorded in results.json; GNNs: reload the saved
``best.pt``), re-predicts the common 501-species test set as a self-check against the stored
``predictions.npz``, then predicts the 78 species of the five homologous series (45 of them
outside the DeepTherm set) and writes one prediction CSV per model:

    <out_dir>/preds_<tag>.csv      smiles, H_kcal, S_cal, Cp300_cal   (DeepTherm units)
    <out_dir>/verify_<tag>.json    max |re-predicted - stored| on the test and validation sets

Feed the CSVs to ``ch2_extrapolation_summary.py`` (increments, MAE from the extrapolated
reference line, C16 errors, in-distribution test MAE) and to
``models/tebnet-v3/check/homologous_controls.py``.

Usage (from the repository root; ~5 min per ML tag on CPU, seconds per GNN tag on GPU):

    python models-baseline/traditional_models/ch2_extrapolation.py --only ml-ridge_prior ml-xgb_physics
    python models-baseline/traditional_models/ch2_extrapolation.py --only gnn-gin_all_L4_d300_sum --device cuda
    python models-baseline/traditional_models/ch2_extrapolation.py --list
"""
from __future__ import annotations
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
BENCH_ROOT = HERE.parent
REPO_ROOT = BENCH_ROOT.parent
TEBNET_V3 = REPO_ROOT / "models" / "tebnet-v3"
for _p in (HERE, TEBNET_V3, REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from baseline_common import GROUPS, dt_to_si, get_split, load_dataset, targets_dt
VERIFY_TOL = 1.0
J_TO_KCAL = 1.0 / 4184.0
J_TO_CAL = 1.0 / 4.184
T_GRID = np.array([300.0, 400.0, 500.0, 600.0, 800.0, 1000.0, 1500.0])
DT_COLS = (["Hf_298.15K_kcal/mol", "S_298.15K_cal/(molK)"]
           + [f"cp_{int(T)}K_cal/(molK)" for T in T_GRID])
GNN_TAGS = {
    "gnn-gin_all_L4_d300_sum": "gin",
    "gnn-gcn_all_L4_d300_sum": "gcn",
    "gnn-gat_all_L4_d300_sum": "gat",
    "gnn-attentivefp_all_L4_d300_attn": "attentivefp",
}
ML_TAGS = {
    "ml-ridge_prior": ("ridge", "prior"),
    "ml-rf_physics": ("rf", "physics"),
    "ml-rf_ecfp-r2b1024": ("rf", "ecfp"),
    "ml-xgb_physics": ("xgb", "physics"),
    "ml-xgb_physics+ecfp-r2b1024": ("xgb", "physics+ecfp"),
}
ALL_TAGS = list(GNN_TAGS) + list(ML_TAGS)
def run_args(extras: dict) -> SimpleNamespace:
    """Argument namespace with the same defaults as ``add_common_args``."""
    base = dict(seed=0, csv=str(BENCH_ROOT / "deeptherm_data.csv"), physics_cache="auto",
                smiles_col="smiles", h_col=None, s_col=None, cp_cols=None, h_unit=None, s_unit=None,
                cp_unit=None, no_3d=False, split_type="random", split_seed=0,
                split_fractions=[0.81, 0.09, 0.10], test_smarts=None, test_group=None, member=0)
    base.update(extras)
    return SimpleNamespace(**base)
def load_homolog_features(series_csv: Path, use_3d: bool = True) -> dict:
    """Physics (v15) features and RDKit mols for the homologous-series SMILES.

    Same code path as ``check/predict_smiles.py``: a temporary CSV with dummy targets is fed to
    ``ThermochemDataset`` so that ``compute_v15_vector`` produces exactly the columns the
    baselines were trained on (no cache exists for these 78 species).
    """
    from tebnet.data.dataset import ThermochemDataset
    from tebnet.data.featurizer import V15_NAMES
    ser = pd.read_csv(series_csv)
    smiles = ser["smiles"].astype(str).str.strip().tolist()
    with tempfile.TemporaryDirectory() as tmp:
        csv = Path(tmp) / "homologs.csv"
        df = pd.DataFrame({"smiles": smiles})
        for c in DT_COLS:
            df[c] = 1.0
        df.to_csv(csv, index=False)
        full = ThermochemDataset.from_csv(csv, T_grid=T_GRID, smiles_col="smiles", physics_cache=None,
                                          h_unit="kcal", s_unit="cal", cp_unit="cal", strict=False,
                                          compute_aux=False, compute_comp=False)
        ds = ThermochemDataset(full.records, T_grid=T_GRID, aux_feature_set="v15", compute_aux=True,
                               use_3d=use_3d)
    dps = ds.datapoints
    if len(dps) != len(smiles):
        raise SystemExit(f"only {len(dps)}/{len(smiles)} series SMILES were featurised")
    return {
        "smiles": [dp.smiles for dp in dps],
        "mols": [dp.mol for dp in dps],
        "X": np.stack([dp.x_physic for dp in dps]).astype(np.float64),
        "feature_names": list(V15_NAMES),
    }
def write_preds(out_dir: Path, tag: str, smiles, h_kcal, s_cal, cp300_cal) -> None:
    pd.DataFrame({"smiles": smiles, "H_kcal": h_kcal, "S_cal": s_cal, "Cp300_cal": cp300_cal}).to_csv(
        out_dir / f"preds_{tag}.csv", index=False, float_format="%.4f")
def verify_run(out_dir: Path, tag: str, run_dir: Path, p_si: np.ndarray, te: np.ndarray, va: np.ndarray) -> dict:
    """Compare re-predicted values with the stored predictions.npz of the same run."""
    z = np.load(run_dir / "predictions.npz", allow_pickle=True)
    rep = {"tag": tag, "run_dir": str(run_dir.relative_to(REPO_ROOT))}
    worst = 0.0
    for split, idx in (("test", te), ("val", va)):
        for key, col in (("h_pred", 0), ("s_pred", 1), ("cp_pred", slice(2, 9))):
            stored = np.asarray(z[f"{split}_{key}"], dtype=np.float64)
            mine = p_si[idx][:, col]
            diff = float(np.nanmax(np.abs(stored - mine)))
            rep[f"max_abs_diff_{split}_{key}"] = diff
            worst = max(worst, diff)
        rep[f"n_{split}"] = int(len(idx))
    rep["max_abs_diff_all"] = worst
    rep["reproduced"] = bool(worst < VERIFY_TOL)
    (out_dir / f"verify_{tag}.json").write_text(json.dumps(rep, indent=2))
    return rep
def refit_ml(tag: str, args_run: SimpleNamespace, data: dict, homologs: dict, tr, va, te,
             results: dict) -> tuple[np.ndarray, np.ndarray]:
    """Refit one ML baseline with the hyper-parameters selected during the original run."""
    from baselines_ml import _Scaled, _XGBMulti, build_features
    Y = targets_dt(data)
    feats = build_features(args_run, data)
    feats_h = build_features(args_run, homologs)
    P = np.full_like(Y, np.nan)
    P_h = np.full((len(homologs["smiles"]), Y.shape[1]), np.nan)
    best = results["best_params"]
    for g, cols in GROUPS.items():
        X, _ = feats[g]
        Xh, _ = feats_h[g]
        ytr, yva = Y[tr][:, cols], Y[va][:, cols]
        keep = np.isfinite(ytr).all(1)
        cfg = best[g]["params"]
        if args_run.model == "ridge":
            from sklearn.linear_model import Ridge
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            model = make_pipeline(StandardScaler(), Ridge(alpha=float(cfg["alpha"])))
            model.fit(X[tr][keep], ytr[keep])
        elif args_run.model == "rf":
            from sklearn.ensemble import RandomForestRegressor
            rcfg = {k: v for k, v in cfg.items() if k != "n_trees"}
            model = _Scaled(RandomForestRegressor(n_estimators=args_run.n_estimators,
                                                  random_state=args_run.seed,
                                                  n_jobs=args_run.n_jobs, **rcfg)).fit(X[tr][keep], ytr[keep])
        else:
            xcfg = {k: v for k, v in cfg.items() if k != "n_trees"}
            model = _XGBMulti(xcfg, args_run.seed, args_run).fit(X[tr][keep], ytr[keep], X[va], yva)
        P[:, cols] = model.predict(X).reshape(len(X), -1)
        P_h[:, cols] = model.predict(Xh).reshape(len(Xh), -1)
    return dt_to_si(P), dt_to_si(P_h)
def reload_gnn(run_dir: Path, args_run: SimpleNamespace, data: dict, homologs: dict, tr, va, te,
               device: str) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from torch_geometric.loader import DataLoader
    from baseline_common import TARGET_NAMES
    from baselines_gnn import TARGET_COLS, build_model, mol_to_graph, predict
    cols = TARGET_COLS[args_run.targets]
    Y = targets_dt(data)
    mu = np.nanmean(Y[tr][:, cols], axis=0)
    sd = np.nanstd(Y[tr][:, cols], axis=0)
    sd[~np.isfinite(sd) | (sd < 1e-12)] = 1.0
    Z = (Y[:, cols] - mu) / sd
    graphs = [mol_to_graph(m, Z[i], i) for i, m in enumerate(data["mols"])]
    in_dim, edge_dim = graphs[0].x.shape[1], graphs[0].edge_attr.shape[1]
    g_hom = [mol_to_graph(m, np.zeros(len(cols)), i) for i, m in enumerate(homologs["mols"])]
    loader_hom = DataLoader(g_hom, batch_size=256, shuffle=False)
    dev = torch.device(device)
    model = build_model(args_run, in_dim, edge_dim, len(cols)).to(dev)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=dev))
    model.eval()
    P = np.full_like(Y, np.nan)
    for k, idx in (("train", tr), ("val", va), ("test", te)):
        loader = DataLoader([graphs[i] for i in idx], batch_size=256, shuffle=False)
        P[idx[:, None], np.asarray(cols)[None, :]] = predict(model, loader, dev, len(Y), len(cols))[idx] * sd + mu
    P_h = np.full((len(homologs["smiles"]), Y.shape[1]), np.nan)
    P_h[:, cols] = predict(model, loader_hom, dev, len(g_hom), len(cols)) * sd + mu
    return dt_to_si(P), dt_to_si(P_h)
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None, help="tags to run (default: all)")
    ap.add_argument("--list", action="store_true", help="list the tags and exit")
    ap.add_argument("--runs_root", type=str, default=str(BENCH_ROOT / "benchmark_results_baselines"))
    ap.add_argument("--series_csv", type=str, default=str(TEBNET_V3 / "check" / "homologous_series.csv"))
    ap.add_argument("--out_dir", type=str, default=str(BENCH_ROOT / "ch2_extrapolation"))
    ap.add_argument("--split_type", default="random")
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="GNN inference device (cpu or cuda)")
    ap.add_argument("--n_jobs", type=int, default=None, help="override the stored n_jobs (XGBoost/RF refits)")
    args = ap.parse_args()
    if args.list:
        for t in ALL_TAGS:
            print(t)
        return
    tags = args.only or ALL_TAGS
    unknown = [t for t in tags if t not in ALL_TAGS]
    if unknown:
        raise SystemExit(f"unknown tags {unknown}; see --list")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[data] DeepTherm set + physics features ...")
    runs_root = Path(args.runs_root)
    data = load_dataset(run_args({"split_type": args.split_type, "split_seed": args.split_seed}))
    tr, va, te, split_name, _ = get_split(run_args({"split_type": args.split_type,
                                                    "split_seed": args.split_seed}), data["smiles"])
    print(f"[data] split {split_name}: {len(tr)} train / {len(va)} val / {len(te)} test")
    print("[data] homologous-series features ...")
    homologs = load_homolog_features(Path(args.series_csv))
    print(f"[data] {len(homologs['smiles'])} series species featurised")
    for tag in tags:
        run_dir = runs_root / tag / split_name / "member0_seed0"
        results = json.loads((run_dir / "results.json").read_text())
        t0 = time.time()
        if tag in ML_TAGS:
            model, features = ML_TAGS[tag]
            overrides = dict(model=model, features=features)
            if args.n_jobs is not None:
                overrides["n_jobs"] = args.n_jobs
            args_run = run_args(dict(results["args"], **overrides))
            p_si, p_hom = refit_ml(tag, args_run, data, homologs, tr, va, te, results)
        else:
            args_run = run_args(results["args"])
            p_si, p_hom = reload_gnn(run_dir, args_run, data, homologs, tr, va, te, args.device)
        rep = verify_run(out_dir, tag, run_dir, p_si, te, va)
        write_preds(out_dir, tag, homologs["smiles"], p_hom[:, 0] * J_TO_KCAL, p_hom[:, 1] * J_TO_CAL,
                    p_hom[:, 2] * J_TO_CAL)
        print(f"[{tag}] reproduced={rep['reproduced']}  max|diff|={rep['max_abs_diff_all']:.3g}  "
              f"({time.time() - t0:.0f} s)  -> preds_{tag}.csv")
if __name__ == "__main__":
    main()

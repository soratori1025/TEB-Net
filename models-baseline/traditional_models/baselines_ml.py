"""Conventional ML baselines for Table 1 under the DeepTherm protocol: ridge, random forest, XGBoost.

Same rows, units and splits as TEB-Net (scripts/baseline_common.py); output readable by
scripts/deeptherm_eval.py and scripts/run_info.py.

One model per property group, i.e. "single-task" like the chemprop single-task runs:
  ΔH_f,298 | S298 | Cp(300…1500 K) (7 outputs; ridge/RF natively multi-output, XGBoost one model per T).

Features (--features):
  physics        all V15 physics columns of TEB-Net (69 with 3-D, from the physics cache)
  prior          per group, exactly the columns of TEB-Net's frozen linear baseline (v15_base_names);
                 with --model ridge this is the "physics prior alone" ablation
  ecfp           Morgan fingerprint (radius 2, 1024 bits as in DeepTherm; --ecfp_counts for counts)
  physics+ecfp   both concatenated

Hyper-parameters are chosen on the VALIDATION set only, with a fixed, identical budget per split:
  ridge   alpha on a 15-point log grid (features standardised on train)
  rf      500 trees; max_features × min_samples_leaf grid (8 configurations)
  xgb     --n_trials configurations drawn once from a fixed grid (same list on every split), each with
          early stopping on the validation MAE (≤ 3000 trees)
The selected model is the one fitted on TRAIN; validation is never used for fitting, test never looked at.

  python scripts/baselines_ml.py --model rf  --features physics --split_seed 0
  python scripts/baselines_ml.py --model xgb --features physics+ecfp --split_type complexity
  python scripts/baselines_ml.py --model ridge --features prior --split_seed 3
"""
from __future__ import annotations
import argparse
import itertools
import time
import numpy as np
from baseline_common import (
    GROUPS, add_common_args, dt_to_si, get_split, load_dataset, pred_dict, prepare_run_dir,
    targets_dt, targets_si, write_run,
)
RIDGE_ALPHAS = np.logspace(-4, 3, 15)
RF_GRID = [{"max_features": mf, "min_samples_leaf": leaf}
           for mf, leaf in itertools.product([1.0, 0.5, 0.3, "sqrt"], [1, 3])]
XGB_SPACE = {
    "learning_rate": [0.02, 0.05, 0.1],
    "max_depth": [3, 5, 7, 9],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.3, 0.5, 0.8, 1.0],
    "reg_lambda": [0.1, 1.0, 10.0],
    "reg_alpha": [0.0, 0.1],
}
def ecfp_matrix(mols, radius: int, n_bits: int, counts: bool) -> np.ndarray:
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    def one(m):
        fp = gen.GetCountFingerprint(m) if counts else gen.GetFingerprint(m)
        arr = np.zeros((n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    return np.stack([one(m) for m in mols])
def build_features(args, data: dict) -> dict[str, tuple[np.ndarray, list[str]]]:
    """Feature matrix (and column names) for each group h / s / cp."""
    from tebnet.data.featurizer import FEATS_3D, v15_base_names
    names = data["feature_names"]
    col = {n: i for i, n in enumerate(names)}
    phys_names = [n for n in names if not (args.no_3d and n in FEATS_3D)]
    use_ecfp = "ecfp" in args.features
    E = ecfp_matrix(data["mols"], args.ecfp_radius, args.ecfp_bits, args.ecfp_counts) if use_ecfp else None
    e_names = [f"ecfp_{i}" for i in range(args.ecfp_bits)] if use_ecfp else []
    out = {}
    for g in GROUPS:
        if args.features == "prior":
            cols = v15_base_names(g, "physics", not args.no_3d)
        elif args.features in ("physics", "physics+ecfp"):
            cols = phys_names
        else:
            cols = []
        X = data["X"][:, [col[n] for n in cols]] if cols else np.zeros((len(data["smiles"]), 0))
        if use_ecfp:
            X = np.hstack([X, E])
        out[g] = (X.astype(np.float64), list(cols) + e_names)
    return out
class _Scaled:
    """Standardises y (per column, train statistics) around a multi-output regressor."""
    def __init__(self, model):
        self.model = model
    def fit(self, X, y):
        self.mu, self.sd = y.mean(0), y.std(0)
        self.sd[self.sd < 1e-12] = 1.0
        z = (y - self.mu) / self.sd
        self.model.fit(X, z.ravel() if z.shape[1] == 1 else z)
        return self
    def predict(self, X):
        p = self.model.predict(X)
        return (p.reshape(len(X), -1) * self.sd) + self.mu
def _mae(pred, true):
    m = np.isfinite(true)
    return float(np.mean(np.abs(pred[m] - true[m])))
def search_ridge(Xtr, ytr, Xva, yva, seed, args):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    trials = []
    for a in RIDGE_ALPHAS:
        m = make_pipeline(StandardScaler(), Ridge(alpha=float(a))).fit(Xtr, ytr)
        trials.append(({"alpha": float(a)}, _mae(m.predict(Xva).reshape(yva.shape), yva), m))
    return trials
def search_rf(Xtr, ytr, Xva, yva, seed, args):
    from sklearn.ensemble import RandomForestRegressor
    trials = []
    for cfg in RF_GRID:
        rf = RandomForestRegressor(n_estimators=args.n_estimators, random_state=seed, n_jobs=args.n_jobs, **cfg)
        m = _Scaled(rf).fit(Xtr, ytr)
        trials.append((dict(cfg), _mae(m.predict(Xva), yva), m))
        print(f"    rf {cfg} -> val MAE {trials[-1][1]:.3f}")
    return trials
def xgb_configs(n_trials: int) -> list[dict]:
    """The same n_trials configurations on every split and every group (fixed seed)."""
    grid = [dict(zip(XGB_SPACE, v)) for v in itertools.product(*XGB_SPACE.values())]
    rng = np.random.default_rng(12345)
    return [grid[i] for i in rng.choice(len(grid), size=min(n_trials, len(grid)), replace=False)]
class _XGBMulti:
    """One XGBRegressor per output column, each early-stopped on its own validation column."""
    def __init__(self, cfg, seed, args):
        self.cfg, self.seed, self.args = cfg, seed, args
        self.models, self.n_trees = [], []
    def fit(self, X, y, Xva, yva):
        from xgboost import XGBRegressor
        for j in range(y.shape[1]):
            ok, okv = np.isfinite(y[:, j]), np.isfinite(yva[:, j])
            m = XGBRegressor(n_estimators=3000, early_stopping_rounds=100, eval_metric="mae",
                             tree_method="hist", device=self.args.xgb_device, random_state=self.seed,
                             n_jobs=self.args.n_jobs, **self.cfg)
            m.fit(X[ok], y[ok, j], eval_set=[(Xva[okv], yva[okv, j])], verbose=False)
            self.models.append(m)
            self.n_trees.append(int(m.best_iteration) + 1)
        return self
    def predict(self, X):
        return np.column_stack([m.predict(X) for m in self.models])
def search_xgb(Xtr, ytr, Xva, yva, seed, args):
    try:
        import xgboost
    except ImportError:
        raise SystemExit("xgboost is not installed:  pip install xgboost")
    trials = []
    for i, cfg in enumerate(xgb_configs(args.n_trials)):
        m = _XGBMulti(cfg, seed, args).fit(Xtr, ytr, Xva, yva)
        c = dict(cfg, n_trees=m.n_trees)
        trials.append((c, _mae(m.predict(Xva), yva), m))
        print(f"    xgb trial {i + 1:>2}/{args.n_trials} {cfg} -> val MAE {trials[-1][1]:.3f}")
    return trials
SEARCH = {"ridge": search_ridge, "rf": search_rf, "xgb": search_xgb}
def main():
    ap = argparse.ArgumentParser(description="Conventional ML baselines under the DeepTherm protocol")
    add_common_args(ap)
    ap.add_argument("--model", choices=sorted(SEARCH), required=True)
    ap.add_argument("--features", choices=["physics", "prior", "ecfp", "physics+ecfp"], required=True)
    ap.add_argument("--ecfp_bits", type=int, default=1024)
    ap.add_argument("--ecfp_radius", type=int, default=2)
    ap.add_argument("--ecfp_counts", action="store_true", help="count fingerprint instead of bits")
    ap.add_argument("--n_estimators", type=int, default=500, help="random-forest trees")
    ap.add_argument("--n_trials", type=int, default=20, help="XGBoost random-search budget")
    ap.add_argument("--xgb_device", default="cpu", help="cpu or cuda")
    ap.add_argument("--n_jobs", type=int, default=-1)
    args = ap.parse_args()
    tag = f"ml-{args.model}_{args.features}"
    if "ecfp" in args.features:
        tag += f"-r{args.ecfp_radius}b{args.ecfp_bits}{'c' if args.ecfp_counts else ''}"
    if args.no_3d:
        tag += "_2d"
    tag += args.tag_suffix
    data = load_dataset(args)
    tr, va, te, split_name, test_smarts = get_split(args, data["smiles"])
    save_dir = prepare_run_dir(args, tag, split_name)
    print(f"[{tag} | {split_name} | member {args.member} | seed {args.seed}] "
          f"{len(tr)} train / {len(va)} val / {len(te)} test")
    Y = targets_dt(data)
    feats = build_features(args, data)
    P = np.full_like(Y, np.nan)
    best, search_log, t0 = {}, {}, time.time()
    for g, cols in GROUPS.items():
        X, names = feats[g]
        ytr, yva = Y[tr][:, cols], Y[va][:, cols]
        keep = np.isfinite(ytr).all(1)
        print(f"  group {g}: {X.shape[1]} features, {keep.sum()} training rows")
        trials = SEARCH[args.model](X[tr][keep], ytr[keep], X[va], yva, args.seed, args)
        cfg, score, model = min(trials, key=lambda t: t[1])
        best[g] = {"params": cfg, "val_mae": score}
        search_log[g] = [{"params": c, "val_mae": s} for c, s, _ in trials]
        print(f"  group {g}: best {cfg}  val MAE {score:.3f}")
        P[:, cols] = model.predict(X).reshape(len(X), -1)
    Ysi, Psi = targets_si(data), dt_to_si(P)
    preds = {k: pred_dict(Psi[i], Ysi[i], data["X"][i], [data["smiles"][j] for j in i])
             for k, i in (("train", tr), ("val", va), ("test", te))}
    write_run(save_dir, args=args, tag=tag, model=f"ML-{args.model}", split_name=split_name,
              test_smarts=test_smarts, data=data, idx={"train": tr, "val": va, "test": te}, preds=preds,
              extra={"best_params": best, "search": search_log, "features": args.features,
                     "n_features": {g: int(feats[g][0].shape[1]) for g in GROUPS},
                     "training_time_sec": round(time.time() - t0, 1)})
if __name__ == "__main__":
    main()

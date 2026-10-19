"""Prior-only screening of the v3 group-additivity columns on WUDILY-CHO (same 5 folds as v1)."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/home/nvsinh1/brats_segmentation/teb-net-model/models-s3/tebnet-v3")
sys.path.insert(0, str(ROOT))
from tebnet.data.featurizer import V15_NAMES, v15_base_names
from tebnet.data.baselines import fit_ridge, fit_cp_cheb_baseline, cp_from_cheb
from tebnet.data.groups import load_or_compute_groups

CSV = "/home/nvsinh1/brats_segmentation/teb-net-model/models-baseline/sota/bruce_benchmark/data/wudily_cho_processed.csv"
FOLDS = "/home/nvsinh1/brats_segmentation/teb-net-model/models-s3/tebnet-v1/results/benchmark_results_wudily/wudily_cho_processed/folds_k5_cv0.json"
df = pd.read_csv(CSV)
cache = pd.read_csv(CSV.replace(".csv", ".physics_v15.csv"))
fd = json.loads(Path(FOLDS).read_text())
smiles = fd["smiles"]
assert smiles == df["smiles"].astype(str).str.strip().tolist()
X15 = cache.set_index("smiles").loc[smiles, V15_NAMES].to_numpy(float)
H = df["Hf_298_kcal_mol"].to_numpy(float)
S = df["S_298_cal_molK"].to_numpy(float)
CPc = [c for c in df.columns if c.startswith("Cp_")]
CP = df[CPc].to_numpy(float)
t0 = time.time()
G, vocab = load_or_compute_groups(CSV, smiles, min_df=3)
print(f"groups: {G.shape[1]} keys ({sum(v.startswith('g:') for v in vocab)} groups, "
      f"{sum(v.startswith('r:') for v in vocab)} ring keys) in {time.time()-t0:.0f}s")
col = {n: i for i, n in enumerate(V15_NAMES)}
folds = [np.array(f) for f in fd["folds"]]


def design(role, use_groups):
    Xb = X15[:, [col[n] for n in v15_base_names(role, "physics", True)]]
    return np.hstack([Xb, G]) if use_groups else Xb


def fit_pred(Xtr, ytr, Xte, lam, min_count, n_base):
    keep = np.ones(Xtr.shape[1], bool)
    if min_count:
        nz = (Xtr[:, n_base:] != 0).sum(0)
        keep[n_base:] = nz >= min_count
    w, b, _ = fit_ridge(Xtr[:, keep], ytr, lam)
    return Xte[:, keep] @ w + b


T = np.array([300, 400, 500, 600, 800, 1000, 1500.0])
configs = [("v15", False, 1e-3, 0)] + [(f"v15+groups mc{mc} lam{lam:g}", True, lam, mc)
                                       for mc in (3, 10, 25) for lam in (1e-4, 1e-3, 1e-2)]
for name, ug, lam, mc in configs:
    res = {"H": [], "S": [], "Cp": []}
    for k, te in enumerate(folds):
        tr = np.sort(np.concatenate([f for j, f in enumerate(folds) if j != k]))
        for prop, y in (("H", H), ("S", S)):
            X = design(prop.lower(), ug)
            nb = X.shape[1] - (G.shape[1] if ug else 0)
            p = fit_pred(X[tr], y[tr], X[te], lam, mc, nb)
            res[prop].append(np.abs(p - y[te]).mean())
        X = design("cp", ug)
        nb = X.shape[1] - (G.shape[1] if ug else 0)
        keep = np.ones(X.shape[1], bool)
        if mc:
            keep[nb:] = (X[tr][:, nb:] != 0).sum(0) >= mc
        bc = fit_cp_cheb_baseline(X[tr][:, keep], CP[tr] * 4.184, T, 1000.0, 300.0, 3000.0, lam_rel=lam)
        cp = cp_from_cheb(X[te][:, keep] @ bc["W"] + bc["b"], T, 1000.0, 300.0, 3000.0) / 4.184
        res["Cp"].append(np.abs(cp - CP[te]).mean())
    print(f"{name:32s} H {np.mean(res['H']):.2f}±{np.std(res['H']):.2f}  S {np.mean(res['S']):.2f}±{np.std(res['S']):.2f}"
          f"  Cp {np.mean(res['Cp']):.2f}", flush=True)

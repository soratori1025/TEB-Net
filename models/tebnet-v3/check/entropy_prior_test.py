"""Entropy prior v2: prior-only test (no network, no GPU; a few minutes, most of it conformer search).

Question: does computing the exact statistical-mechanics parts of S°298 (translation, rigid rotation of the
lowest-energy conformer, −R ln σ, R ln(2S+1)) and fitting only the additive remainder give a better S prior
than v15, and does it keep the entropy increment per CH2 on long chains? Variants (tebnet.data.featurizer.
s_prior_columns):
    v1          published v15 S prior
    v1_minconf  v1 with ln I of the lowest-energy conformer (isolates the conformer effect)
    v2_free     exact-term columns + additive counts, all fitted (do the data follow the conventions?)
    v2_rot      S_trans and S_rot fixed (coefficient R), σ and spin fitted
    v2_exact    S_trans, S_rot, −ln σ and ln(2S+1) all fixed

Decision rule, fixed BEFORE looking at test data (DeepTherm random split 0 and complexity split, prior only):
    GO   if the best v2 variant (lowest mean validation MAE over the two splits) has a lower validation S MAE
         than v1 on BOTH splits; that variant is then used for the S/Cp models.
    NO-GO otherwise: keep v1 and report the entropy limitation as it is.
Validation numbers are always printed. Test-set numbers and the homologous-series extrapolation are printed
only with --show_test; run that once, after the decision, and report it whatever it shows.

Also writes the prior-only predictions for the homologous-series controls (check/homologous_controls.py):
    preds_prior_v1.csv          H_kcal, S_cal, Cp300_cal of the published v15 priors (random split 0)
    preds_prior_S_<variant>.csv S_cal of each S-prior variant

Usage (from the repository root, like check/predict_smiles.py)
  python check/entropy_prior_test.py --csv models-baseline/deeptherm_data.csv \
      --series check/homologous_series.csv --workers 16            # decision (validation only)
  python check/entropy_prior_test.py ... --show_test                 # once, after the decision
"""
from __future__ import annotations

import argparse
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

R_GAS = 8.314462618
J_TO_KCAL = 1.0 / 4184.0
J_TO_CAL = 1.0 / 4.184


def setup_paths():
    here = Path(__file__).resolve().parent
    for p in (here, here.parent / "scripts"):
        if (p / "train.py").exists():
            sys.path.insert(0, str(p))
            sys.path.insert(0, str(here))
            return
    raise SystemExit("train.py not found next to this script or in ../scripts")


def import_split():
    for mod in ("tebnet.data.splits", "tebnet.splits", "train"):
        try:
            return __import__(mod, fromlist=["deeptherm_split"]).deeptherm_split
        except (ImportError, AttributeError):
            continue
    raise SystemExit("deeptherm_split not found (tebnet.data.splits / train.py)")


def entropy_terms(smiles: list[str], cache: Path, n_conf: int, seed: int, workers: int) -> pd.DataFrame:
    """ENTROPY_V2_NAMES for each SMILES, cached in `cache` (recomputed when the version or n_conf differ)."""
    from tebnet.data.featurizer import ENTROPY_V2_NAMES, ENTROPY_V2_VERSION, entropy_v2_from_smiles
    old = pd.DataFrame()
    if cache.exists():
        old = pd.read_csv(cache)
        if (old.get("version", pd.Series(dtype=str)).astype(str) != ENTROPY_V2_VERSION).any() or \
                (old.get("n_conf", pd.Series(dtype=float)) != n_conf).any():
            print(f"  {cache}: other version or n_conf, recomputing")
            old = pd.DataFrame()
    have = set(old["smiles"].astype(str)) if not old.empty else set()
    todo = sorted({s for s in smiles if s not in have})
    if todo:
        print(f"  conformer search for {len(todo)} species ({n_conf} ETKDG + 1 all-anti conformer each, "
              f"{workers} workers) ...", flush=True)
        fn = partial(entropy_v2_from_smiles, n_conf=n_conf, seed=seed)
        if workers > 1:
            with Pool(workers) as pool:
                res = pool.map(fn, todo, chunksize=8)
        else:
            res = [fn(s) for s in todo]
        new = pd.DataFrame([r for r in res if r is not None])
        new["version"], new["n_conf"] = ENTROPY_V2_VERSION, n_conf
        old = pd.concat([old, new], ignore_index=True) if not old.empty else new
        old.to_csv(cache, index=False, float_format="%.6f")
        print(f"  wrote {cache}")
    df = old.drop_duplicates("smiles").set_index("smiles")
    out = df.reindex(smiles)
    bad = out["S_trans_R"].isna().sum()
    if bad:
        print(f"  warning: {bad} SMILES without entropy terms (invalid SMILES?)")
    print("  force field used:", out["ff"].value_counts(dropna=False).to_dict(),
          f"| failed embeddings: {int(out['moi2_invalid'].fillna(1).sum())}")
    return out[ENTROPY_V2_NAMES].astype(float)


def mae(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.abs(a[m] - b[m]).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="DeepTherm data CSV (as used by train.py)")
    ap.add_argument("--smiles_col", default="smiles")
    ap.add_argument("--series", default=None, help="homologous_series.csv (for the prior-only controls)")
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--ridge_lambda", type=float, default=1e-3)
    ap.add_argument("--n_conf", type=int, default=10)
    ap.add_argument("--conf_seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--variants", nargs="+", default=None)
    ap.add_argument("--show_test", action="store_true")
    ap.add_argument("--out_dir", default=".")
    args = ap.parse_args()

    setup_paths()
    from train import DEEPTHERM_T_GRID                    # also sets the project paths
    from tebnet.data.dataset import ThermochemDataset
    from tebnet.data.featurizer import (V15_NAMES, ENTROPY_V2_NAMES, S_PRIOR_VARIANTS, s_prior_columns,
                                        v15_base_names, compute_v15_vector, entropy_v2_cache_path)
    from tebnet.data.baselines import (fit_scalar_baseline, fit_scalar_baseline_fixed, fit_cp_cheb_baseline,
                                       cp_from_cheb)
    deeptherm_split = import_split()
    variants = args.variants or list(S_PRIOR_VARIANTS)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_grid = np.asarray(DEEPTHERM_T_GRID, dtype=np.float64)
    t_mid, t_lo, t_hi = 1000.0, float(t_grid[t_grid <= 1000.0].min()), 3000.0
    ds = ThermochemDataset.from_csv(args.csv, T_grid=t_grid, smiles_col=args.smiles_col, physics_cache="auto",
                                    strict=True, compute_aux=True, aux_feature_set="v15", use_3d=True,
                                    compute_comp=False)
    dps = ds.datapoints
    smiles = [dp.smiles for dp in dps]
    print(f"{Path(args.csv).name}: {len(dps)} species")
    ent = entropy_terms(smiles, entropy_v2_cache_path(args.csv), args.n_conf, args.conf_seed, args.workers)
    names = list(V15_NAMES) + list(ENTROPY_V2_NAMES)
    col = {n: i for i, n in enumerate(names)}
    X = np.hstack([np.stack([dp.x_physic for dp in dps]).astype(np.float64), ent.to_numpy(np.float64)])
    H = np.array([dp.dh_f_298 for dp in dps], dtype=np.float64)
    S = np.array([dp.s_298 for dp in dps], dtype=np.float64)
    CP = np.stack([dp.cp_grid for dp in dps]).astype(np.float64)

    def fit_s(variant, idx):
        cols, fixed_R = s_prior_columns(variant)
        ci = [col[c] for c in cols]
        if fixed_R:
            fx = np.array([fixed_R.get(c, np.nan) * R_GAS for c in cols])
            b = fit_scalar_baseline_fixed(X[idx][:, ci], S[idx], fx, args.ridge_lambda)
        else:
            b = fit_scalar_baseline(X[idx][:, ci], S[idx], args.ridge_lambda)
        return cols, ci, b

    # ── validation (and test) MAE of the S prior on the two DeepTherm splits ──
    rows, coef_rows = [], []
    for strategy in ("random", "complexity"):
        tr, va, te = (np.asarray(a) for a in deeptherm_split(smiles, strategy, seed=args.split_seed))
        for v in variants:
            cols, ci, b = fit_s(v, tr)
            pred = X[:, ci] @ b["w"] + b["b"]
            r = {"split": f"{strategy} {args.split_seed}", "variant": v,
                 "train": mae(pred[tr], S[tr]) * J_TO_CAL, "val": mae(pred[va], S[va]) * J_TO_CAL}
            if args.show_test:
                r["test"] = mae(pred[te], S[te]) * J_TO_CAL
            rows.append(r)
            if strategy == "random":
                w = dict(zip(cols, b["w"] / R_GAS))
                coef_rows.append({"variant": v, **{k: w.get(k, np.nan) for k in w.keys() if abs(w.get(k, 0)) > 1e-4}})
    res = pd.DataFrame(rows)
    print("\n### S°298 prior only: MAE in cal/(mol K)\n")
    print(res.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n### Coefficients of the statistical-mechanics terms, in units of R (random split; physical values: "
          "S_trans +1, S_rot +1, ½ln I +1, ln σ −1, ln(2S+1) +1, chirality +ln 2 = +0.69). Fixed ones show the "
          "fixed value.\n")
    print(pd.DataFrame(coef_rows).to_string(index=False, float_format=lambda x: f"{x:+.2f}"))

    # ── decision ──
    v2 = [v for v in variants if v.startswith("v2_") and v != "v2_free"]
    if "v1" in variants and v2:
        val = res.pivot(index="variant", columns="split", values="val")
        best = val.loc[v2].mean(axis=1).idxmin()
        go = bool((val.loc[best] < val.loc["v1"]).all())
        print(f"\nDecision (validation only): best v2 variant = {best}; "
              + ", ".join(f"{c}: v1 {val.loc['v1', c]:.3f} -> {val.loc[best, c]:.3f}" for c in val.columns)
              + f"  =>  {'GO' if go else 'NO-GO'}")

    # ── prior-only predictions on the homologous series (priors fitted on random split 0) ──
    if args.series:
        ser = pd.read_csv(args.series)
        ssmi = ser["smiles"].astype(str).str.strip().tolist()
        from rdkit import Chem
        Xs15 = np.stack([compute_v15_vector(Chem.MolFromSmiles(s), use_3d=True) for s in ssmi]).astype(np.float64)
        ent_s = entropy_terms(ssmi, entropy_v2_cache_path(args.series), args.n_conf, args.conf_seed, args.workers)
        Xs = np.hstack([Xs15, ent_s.to_numpy(np.float64)])
        tr, _, _ = (np.asarray(a) for a in deeptherm_split(smiles, "random", seed=args.split_seed))
        bh_cols = [col[c] for c in v15_base_names("h", "physics", True)]
        bc_cols = [col[c] for c in v15_base_names("cp", "physics", True)]
        bh = fit_scalar_baseline(X[tr][:, bh_cols], H[tr], args.ridge_lambda)
        bc = fit_cp_cheb_baseline(X[tr][:, bc_cols], CP[tr], t_grid, t_mid, t_lo, t_hi, lam_rel=args.ridge_lambda)
        _, ci1, bs1 = fit_s("v1", tr)
        cp300 = cp_from_cheb(Xs[:, bc_cols] @ bc["W"] + bc["b"], np.array([300.0]), t_mid, t_lo, t_hi)[:, 0]
        p1 = pd.DataFrame({"smiles": ssmi, "H_kcal": (Xs[:, bh_cols] @ bh["w"] + bh["b"]) * J_TO_KCAL,
                           "S_cal": (Xs[:, ci1] @ bs1["w"] + bs1["b"]) * J_TO_CAL, "Cp300_cal": cp300 * J_TO_CAL})
        p1.to_csv(out_dir / "preds_prior_v1.csv", index=False, float_format="%.3f")
        spec = []
        for v in [v for v in variants if v != "v1"]:          # v1 is in preds_prior_v1.csv
            _, ci, b = fit_s(v, tr)
            f = out_dir / f"preds_prior_S_{v}.csv"
            pd.DataFrame({"smiles": ssmi, "S_cal": (Xs[:, ci] @ b["w"] + b["b"]) * J_TO_CAL}).to_csv(
                f, index=False, float_format="%.3f")
            spec.append(f"prior {v}={f}")
        print(f"\nwrote {out_dir / 'preds_prior_v1.csv'} and preds_prior_S_<variant>.csv")
        if args.show_test:
            from homologous_controls import load_models, report
            print("\n### Homologous series, prior only (test of extrapolation; priors from random split "
                  f"{args.split_seed})")
            report(ser, load_models(ser, [f"prior v1={out_dir / 'preds_prior_v1.csv'}"] + spec),
                   props=("S", "H", "Cp300"))
        else:
            print("homologous extrapolation not shown (validation-only run); rerun with --show_test after deciding")


if __name__ == "__main__":
    main()
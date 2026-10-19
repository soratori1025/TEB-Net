"""Size extrapolation on homologous series: TEB-Net against DeepTherm's published predictions (no training).

Runs trained TEB-Net checkpoints (decoupled ΔH and S/Cp models, ensembles allowed) on any list of SMILES and,
for homologous series, compares the increment per CH2 with the training-range increment and with reference
values when given. Input: homologous_series.csv (columns series, n_carbon, smiles, in_deeptherm_set,
data_* = values in the DeepTherm set, deeptherm_* = DeepTherm Supplementary Data 4, ref_* = experiment).

Reference line ("dataset increment"): for each series, a straight line in n fitted to the members that are in
the DeepTherm set with n >= 3, extrapolated to larger n. Long-chain homologues are known to be linear in n to
within a few tenths of a kcal/mol per CH2, so the line is a physical reference, not a measured value; the
ref_* columns (e.g. NIST) are the measured check.

Usage
  B=benchmark_results_v153
  python check/predict_smiles.py --smiles_csv homologous_series.csv \
      --h_runs   $B/thermo_physics_chebyshev_3d_grouped_d5_global_w1-0-0_mon-h/random0/member*_seed* \
      --scp_runs $B/thermo_physics_chebyshev_3d_grouped_d5_global_w0-1.5-1_mon-scp/random0/member*_seed* \
      --out homologous_predictions.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

J_TO_KCAL = 1.0 / 4184.0
J_TO_CAL = 1.0 / 4.184
DT_COLS = ["Hf_298.15K_kcal/mol", "S_298.15K_cal/(molK)", "cp_300K_cal/(molK)", "cp_400K_cal/(molK)",
           "cp_500K_cal/(molK)", "cp_600K_cal/(molK)", "cp_800K_cal/(molK)", "cp_1000K_cal/(molK)",
           "cp_1500K_cal/(molK)"]


def setup_paths():
    here = Path(__file__).resolve().parent
    for p in (here, here.parent / "scripts"):
        if (p / "train.py").exists():
            sys.path.insert(0, str(p))
            return
    raise SystemExit("train.py not found next to this script or in ../scripts")


def predict(run_dirs, smiles, batch_size=64, cache_path=None):
    """Weighted ensemble over run_dirs (weights 1/validation MAE, as deeptherm_eval.py). Returns dict of arrays
    indexed like `smiles` (NaN where RDKit or the featurizer rejected a SMILES) and the per-run weights."""
    setup_paths()
    import torch
    from torch.utils.data import DataLoader
    from train import make_atom_featurizer, morgan_bits            # also sets the project paths
    from tebnet.data.dataset import ThermochemDataset
    from tebnet.data.collate import ThermoCollater
    from tebnet.models.tebnet import TEBNetV20

    cache, preds, vals = {}, [], []
    for rd in run_dirs:
        rd = Path(rd)
        res = json.loads((rd / "results.json").read_text())
        a = res["args"]
        t_grid = np.asarray(res["t_grid"], dtype=np.float64)
        use_3d = not a.get("no_3d", False)
        s_prior = a.get("s_prior", "v1") if a.get("baseline", "physics") != "none" else "v1"
        key = (a["atom_featurizer"], use_3d, a.get("ecfp_bits", 0), s_prior, a.get("n_conf", 10))
        if key not in cache:
            with tempfile.TemporaryDirectory() as tmp:
                csv = Path(tmp) / "query.csv"
                df = pd.DataFrame({"SMILES": smiles})
                for c in DT_COLS:
                    df[c] = 1.0                                   # dummy targets; only the inputs are used
                df.to_csv(csv, index=False)
                full = ThermochemDataset.from_csv(csv, T_grid=t_grid, smiles_col="SMILES", physics_cache=None,
                                                  h_unit="kcal", s_unit="cal", cp_unit="cal",
                                                  strict=False, compute_aux=False, compute_comp=False)
                ds = ThermochemDataset(full.records, T_grid=t_grid, aux_feature_set="v15", compute_aux=True,
                                       use_3d=use_3d)
            if s_prior != "v1":                                   # same column order as train.py
                from tebnet.data.featurizer import load_or_compute_entropy_v2
                ent = load_or_compute_entropy_v2([dp.smiles for dp in ds.datapoints], cache_path, a.get("n_conf", 10),
                                                 workers=a.get("conf_workers", 8))
                for dp, e in zip(ds.datapoints, ent):
                    dp.x_physic = np.concatenate([dp.x_physic, e]).astype(np.float32)
            if a.get("ecfp_bits", 0) > 0:
                for dp in ds.datapoints:
                    dp.x_physic = np.concatenate([dp.x_physic, morgan_bits(dp.mol, a["ecfp_bits"])]).astype(np.float32)
            cache[key] = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                    collate_fn=ThermoCollater(atom_featurizer=make_atom_featurizer(a["atom_featurizer"])))
        ckpt = Path(res.get("checkpoint") or "")
        if not ckpt.exists():
            ckpt = rd / "best.ckpt"
        model = TEBNetV20.load_from_checkpoint(str(ckpt), map_location="cpu").eval()
        out = {"smiles": [], "h": [], "s": [], "cp": []}
        with torch.no_grad():
            for batch in cache[key]:
                o = model(batch)
                out["smiles"].extend(batch.smiles)
                out["h"].append(o["dh_f_298"].double().numpy() * J_TO_KCAL)
                out["s"].append(o["s_298"].double().numpy() * J_TO_CAL)
                out["cp"].append(o["cp"].double().numpy() * J_TO_CAL)
        pos = {s: i for i, s in enumerate(smiles)}
        idx = [pos.get(s, -1) for s in out["smiles"]]
        if min(idx, default=0) < 0:
            raise SystemExit("the dataset changed some SMILES strings; pass canonical SMILES in the input CSV")
        p = {"h": np.full(len(smiles), np.nan), "s": np.full(len(smiles), np.nan),
             "cp": np.full((len(smiles), len(t_grid)), np.nan)}
        p["h"][idx] = np.concatenate(out["h"])
        p["s"][idx] = np.concatenate(out["s"])
        p["cp"][idx] = np.concatenate(out["cp"])
        preds.append(p)
        v = res["val"]
        vals.append({"h": v["H298_kcal"]["mae"],
                     "scp": (v["S298_cal"]["mae"] + len(t_grid) * v["Cp_mean_MAE_cal"]) / (1 + len(t_grid))})
        print(f"  {rd}: {len(out['smiles'])}/{len(smiles)} molecules")
    return preds, vals, t_grid


def weighted(preds, vals, key, wkey):
    w = 1.0 / np.array([v[wkey] for v in vals])
    w = w / w.sum()
    return np.tensordot(w, np.stack([p[key] for p in preds]), axes=(0, 0))


def series_report(df, prop="H", unit="kcal/mol"):
    """Per series: increment per CH2 and errors beyond the DeepTherm set, for prop in H, S, Cp300."""
    suf = "kcal" if prop == "H" else "cal"
    data, teb, dt, ref = f"data_{prop}_{suf}", f"teb_{prop}_{suf}", f"deeptherm_{prop}_{suf}", f"ref_{prop}_{suf}"
    print(f"\n### Homologous series: {prop} ({unit})\n")
    print("| Series | n beyond the set | Increment per CH2 in the set | TEB-Net increment | DeepTherm increment "
          "| MAE vs line: TEB-Net | MAE vs line: DeepTherm | n with reference | MAE vs reference: TEB-Net / DeepTherm |")
    print("|---|---|---|---|---|---|---|---|---|")
    lines = []
    for name, g in df.groupby("series", sort=False):
        g = g.sort_values("n_carbon")
        fit = g[(g.in_deeptherm_set == 1) & (g.n_carbon >= 3) & g[data].notna()]
        if len(fit) < 3:
            continue
        slope, icpt = np.polyfit(fit.n_carbon, fit[data], 1)
        out = g[g.in_deeptherm_set == 0].copy()
        if out.empty:
            continue
        line = slope * out.n_carbon + icpt
        out[f"line_{prop}_{suf}"] = line
        lines.append(out[["series", "n_carbon", f"line_{prop}_{suf}"]])

        def inc(col):
            if col not in out:
                return np.nan
            o = out[out[col].notna()]
            return np.polyfit(o.n_carbon, o[col], 1)[0] if len(o) >= 2 else np.nan

        def mae(col, r):
            if col not in out:
                return np.nan
            m = out[col].notna() & r.notna()
            return np.abs(out.loc[m, col] - r[m]).mean() if m.any() else np.nan

        r = out[ref] if ref in out else pd.Series(np.nan, index=out.index)
        n_ref = int(r.notna().sum())
        print(f"| {name} | {out.n_carbon.min()}–{out.n_carbon.max()} | {slope:+.2f} | {inc(teb):+.2f} | "
              f"{inc(dt):+.2f} | {mae(teb, line):.2f} | {mae(dt, line):.2f} | "
              f"{n_ref} | " + (f"{mae(teb, r):.2f} / {mae(dt, r):.2f} |" if n_ref else "– |"))
    if prop == "H":
        print("\nIncrement = slope against n over the members outside the DeepTherm set; line = straight line fitted "
              "to the members inside the set (n >= 3) and extrapolated.")
    return pd.concat(lines) if lines else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles_csv", required=True)
    ap.add_argument("--h_runs", nargs="+", required=True)
    ap.add_argument("--scp_runs", nargs="+", default=None)
    ap.add_argument("--out", default="homologous_predictions.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.smiles_csv)
    smiles = df.smiles.astype(str).tolist()
    cache_path = Path(args.smiles_csv).with_name(f"{Path(args.smiles_csv).stem}.entropy_v2.csv")
    if not cache_path.exists():
        cache_path = None
    print("ΔH models:")
    ph, vh, t_grid = predict(args.h_runs, smiles, cache_path=cache_path)
    if args.scp_runs:
        print("S/Cp models:")
        ps, vs, _ = predict(args.scp_runs, smiles, cache_path=cache_path)
    else:
        ps, vs = ph, vh
    df["teb_H_kcal"] = weighted(ph, vh, "h", "h")
    df["teb_S_cal"] = weighted(ps, vs, "s", "scp")
    cp = weighted(ps, vs, "cp", "scp")
    for j, T in enumerate(t_grid):
        if T in (300.0, 1500.0):
            df[f"teb_Cp{T:.0f}_cal"] = cp[:, j]
    if "series" in df and "n_carbon" in df:
        for prop, unit in (("H", "kcal/mol"), ("S", "cal/(mol K)"), ("Cp300", "cal/(mol K)")):
            lines = series_report(df, prop, unit)
            if lines is not None:
                df = df.merge(lines, on=["series", "n_carbon"], how="left")
    df.to_csv(args.out, index=False, float_format="%.3f")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
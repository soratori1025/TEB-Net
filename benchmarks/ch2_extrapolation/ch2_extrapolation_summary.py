"""Summarise the CH2 extrapolation benchmark: increments, reference-line deviations, C16 errors.

Reads the prediction CSVs written by ``ch2_extrapolation.py`` / ``ch2_extrapolation_dmpnn.py``
plus the stored TEB-Net and DeepTherm columns of the homologous series, and reports for every
model and property (textbook convention, identical to models/tebnet-v3/check/homologous_controls.py):

  * increment per CH2 -- slope fitted to the members OUTSIDE the DeepTherm set;
  * MAE from the line -- mean |prediction - line| over those members, the line being fitted to the
    in-set members with n >= 3 and extrapolated;
  * error at C16 (largest member of each series);
  * recovered fraction = model increment / reference increment;

plus, per model, the in-distribution DeepTherm random0 test MAE of the same model, for the
"accuracy in distribution does not predict extrapolation" comparison.

    python models-baseline/traditional_models/ch2_extrapolation_summary.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE
REPO_ROOT = HERE.parent.parent
BENCH_ROOT = REPO_ROOT / "models-baseline"
SERIES = REPO_ROOT / "models" / "tebnet-v3" / "check" / "homologous_series.csv"
MEMBERS = REPO_ROOT / "tebnet_manuscript" / "01_Manuscript" / "figures" / "homologous_members.csv"
PREDICTIONS = REPO_ROOT / "models" / "tebnet-v3" / "homologous_predictions.csv"
RUNS = BENCH_ROOT / "benchmark_results_baselines"
NCON = {"H": "kcal", "S": "cal", "Cp300": "cal"}
SERIES_ORDER = ["n-alkane", "1-alkyl radical", "1-alkyl hydroperoxide", "1-alkene", "n-aldehyde"]
SERIES_LABEL = {"n-alkane": "n-alkanes", "1-alkyl radical": "1-alkyl radicals",
                "1-alkyl hydroperoxide": "1-alkyl hydroperoxides", "1-alkene": "1-alkenes",
                "n-aldehyde": "n-aldehydes"}
MODELS: dict[str, dict] = {
    "DeepTherm (pretrained)": dict(cls="Published baseline", kind="col", prefix="deeptherm_", test_mae=1.25),
    "TEB-Net (trimmed mean, 10)": dict(cls="This work", kind="csv", path=PREDICTIONS, col_prefix="trim_", test_mae=1.267),
    "TEB-Net (member 0)": dict(cls="This work", kind="members", prefix="m0_", test_mae=1.41),
    "D-MPNN (chemprop)": dict(cls="Message-passing GNN", kind="csv", path=OUT_DIR / "preds_dmpnn_chemprop.csv"),
    "GIN": dict(cls="Message-passing GNN", kind="csv", path=OUT_DIR / "preds_gnn-gin_all_L4_d300_sum.csv"),
    "GCN": dict(cls="Message-passing GNN", kind="csv", path=OUT_DIR / "preds_gnn-gcn_all_L4_d300_sum.csv"),
    "GAT": dict(cls="Message-passing GNN", kind="csv", path=OUT_DIR / "preds_gnn-gat_all_L4_d300_sum.csv"),
    "AttentiveFP": dict(cls="Message-passing GNN", kind="csv", path=OUT_DIR / "preds_gnn-attentivefp_all_L4_d300_attn.csv"),
    "Random forest (physics)": dict(cls="Conventional ML", kind="csv", path=OUT_DIR / "preds_ml-rf_physics.csv"),
    "Random forest (ECFP)": dict(cls="Conventional ML", kind="csv", path=OUT_DIR / "preds_ml-rf_ecfp-r2b1024.csv"),
    "XGBoost (physics)": dict(cls="Conventional ML", kind="csv", path=OUT_DIR / "preds_ml-xgb_physics.csv"),
    "XGBoost (physics+ECFP)": dict(cls="Conventional ML", kind="csv", path=OUT_DIR / "preds_ml-xgb_physics+ecfp-r2b1024.csv"),
    "Ridge (additive prior)": dict(cls="Conventional ML", kind="csv", path=OUT_DIR / "preds_ml-ridge_prior.csv"),
}
TAG_OF = {
    "D-MPNN (chemprop)": None,
    "GIN": "gnn-gin_all_L4_d300_sum", "GCN": "gnn-gcn_all_L4_d300_sum", "GAT": "gnn-gat_all_L4_d300_sum",
    "AttentiveFP": "gnn-attentivefp_all_L4_d300_attn",
    "Random forest (physics)": "ml-rf_physics", "Random forest (ECFP)": "ml-rf_ecfp-r2b1024",
    "XGBoost (physics)": "ml-xgb_physics", "XGBoost (physics+ECFP)": "ml-xgb_physics+ecfp-r2b1024",
    "Ridge (additive prior)": "ml-ridge_prior",
}
def load_series() -> pd.DataFrame:
    ser = pd.read_csv(SERIES)
    ser["smiles"] = ser["smiles"].astype(str).str.strip()
    return ser
def model_predictions(label: str, ser: pd.DataFrame) -> dict[str, pd.Series]:
    """Per-property prediction series aligned with ``ser`` rows (NaN where unavailable)."""
    spec = MODELS[label]
    out = {}
    if spec["kind"] == "col":
        for p, suf in NCON.items():
            c = f"{spec['prefix']}{p}_{suf}"
            out[p] = ser[c].astype(float) if c in ser.columns else pd.Series(np.nan, index=ser.index)
    elif spec["kind"] == "members":
        mem = pd.read_csv(MEMBERS)
        mem = mem.assign(smiles=mem["smiles"].astype(str).str.strip()).drop_duplicates("smiles")
        for p, suf in NCON.items():
            vals = ser["smiles"].map(mem.set_index("smiles")[f"{spec['prefix']}{p}"])
            out[p] = pd.Series(vals.to_numpy(float), index=ser.index)
    else:
        path = Path(spec["path"])
        if not path.exists():
            print(f"  note: {label}: {path.name} missing - all values NaN")
            return {p: pd.Series(np.nan, index=ser.index) for p in NCON}
        df = pd.read_csv(path).assign(smiles=lambda d: d["smiles"].astype(str).str.strip())
        df = df.drop_duplicates("smiles").set_index("smiles")
        pre = spec.get("col_prefix", "")
        for p, suf in NCON.items():
            vals = ser["smiles"].map(df[f"{pre}{p}_{suf}"])
            out[p] = pd.Series(vals.to_numpy(float), index=ser.index)
    return out
def per_series(ser: pd.DataFrame, pred: pd.Series, prop: str) -> list[dict]:
    """Increment, MAE from the extrapolated line and end-of-series error, per series."""
    rows = []
    for name in SERIES_ORDER:
        g = ser[ser["series"] == name]
        data_col = {"H": "data_H_kcal", "S": "data_S_cal", "Cp300": "data_Cp300_cal"}[prop]
        fit = g[(g["in_deeptherm_set"] == 1) & (g["n_carbon"] >= 3) & g[data_col].notna()]
        slope, icpt = np.polyfit(fit["n_carbon"], fit[data_col], 1)
        out = g[g["in_deeptherm_set"] == 0]
        line = slope * out["n_carbon"] + icpt
        p = pred.loc[out.index]
        ok = np.isfinite(p.to_numpy(float))
        if ok.sum() < 2:
            rows.append(dict(series=name, line_increment=float(slope), increment=np.nan, recovered=np.nan,
                             mae_line=np.nan, n_out=int(ok.sum()), end_n=np.nan, end_error=np.nan))
            continue
        inc = float(np.polyfit(out["n_carbon"][ok], p[ok], 1)[0])
        row = dict(series=name, line_increment=float(slope), increment=inc,
                   recovered=inc / slope if np.isfinite(inc) else np.nan,
                   mae_line=float(np.abs(p[ok] - line[ok]).mean()), n_out=int(ok.sum()))
        k = int(np.argmax(out["n_carbon"][ok].to_numpy()))
        row["end_n"] = int(out["n_carbon"][ok].to_numpy()[k])
        row["end_error"] = float(abs(p[ok].to_numpy()[k] - line[ok].to_numpy()[k]))
        rows.append(row)
    return rows
def in_distribution_mae(label: str) -> dict:
    tag = TAG_OF.get(label)
    if label.startswith("D-MPNN"):
        v = OUT_DIR / "verify_dmpnn_chemprop.json"
        return {"test_mae_H": json.loads(v.read_text())["test_mae_H"]} if v.exists() else {"test_mae_H": np.nan}
    if tag is None:
        return {"test_mae_H": MODELS[label]["test_mae"]}
    p = RUNS / tag / "random0" / "member0_seed0" / "results.json"
    if not p.exists():
        return {"test_mae_H": np.nan}
    r = json.loads(p.read_text())
    return {"test_mae_H": r["test"]["H298_kcal"]["mae"], "val_mae_mean9": r["val_deeptherm_mae"]}
def main() -> None:
    ser = load_series()
    long_rows, model_rows = [], []
    for label, spec in MODELS.items():
        preds = model_predictions(label, ser)
        for prop in ("H", "S", "Cp300"):
            for r in per_series(ser, preds[prop], prop):
                long_rows.append(dict(model=label, cls=spec["cls"], property=prop, **r))
        incl = in_distribution_mae(label)
        row = dict(model=label, cls=spec["cls"], **incl)
        for prop in ("H", "S", "Cp300"):
            sub = [x for x in long_rows if x["model"] == label and x["property"] == prop]
            row[f"increment_{prop}"] = (f"{np.nanmin([x['increment'] for x in sub]):+.2f} to "
                                        f"{np.nanmax([x['increment'] for x in sub]):+.2f}")
            row[f"mean_mae_line_{prop}"] = float(np.nanmean([x["mae_line"] for x in sub]))
        h = [x for x in long_rows if x["model"] == label and x["property"] == "H"]
        row["mean_mae_line_H"] = float(np.nanmean([x["mae_line"] for x in h]))
        row["mean_end_error_H"] = float(np.nanmean([x["end_error"] for x in h]))
        model_rows.append(row)
    long_df = pd.DataFrame(long_rows)
    mod_df = pd.DataFrame(model_rows)
    long_df.to_csv(OUT_DIR / "ch2_extrapolation_series.csv", index=False, float_format="%.4f")
    mod_df.to_csv(OUT_DIR / "ch2_extrapolation_models.csv", index=False, float_format="%.4f")
    sub = mod_df.dropna(subset=["test_mae_H", "mean_mae_line_H"])
    rx = pd.Series(sub["test_mae_H"].to_numpy(float)).rank().to_numpy()
    ry = pd.Series(sub["mean_mae_line_H"].to_numpy(float)).rank().to_numpy()
    rho = float(np.corrcoef(rx, ry)[0, 1])
    stats = {"n_models": int(len(sub)), "spearman_rho_indist_vs_extrap": rho}
    (OUT_DIR / "ch2_extrapolation_stats.json").write_text(json.dumps(stats, indent=2))
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 50)
    print("\n== per-model summary ==")
    print(mod_df[["model", "cls", "test_mae_H", "increment_H", "mean_mae_line_H", "mean_end_error_H"]]
          .to_string(index=False))
    print("\n== per-series enthalpy detail ==")
    print(long_df[long_df["property"] == "H"][["model", "series", "line_increment", "increment",
                                                "recovered", "mae_line", "end_error"]].to_string(index=False))
    print(f"\nSpearman rho (in-distribution H MAE vs mean MAE from the line, {len(sub)} models): {rho:.3f}")
    print(f"\nwrote {OUT_DIR}/ch2_extrapolation_series.csv and ch2_extrapolation_models.csv")
if __name__ == "__main__":
    main()

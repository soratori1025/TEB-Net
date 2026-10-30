"""Controls on the homologous series (Reviewer #2, point 2): which part of a model keeps the increment per CH2?

Compares any number of models on the five homologous series of the paper: for each series and property, the
increment per CH2 over the members OUTSIDE the DeepTherm set, the MAE of those members from the straight line
fitted to the members inside the set (n >= 3), and the error for n-hexadecane against NIST. Pure pandas/numpy
(no RDKit, no torch), so it runs anywhere.

Inputs
  --series   homologous_series.csv, or the homologous_predictions.csv written by predict_smiles.py (then the
             TEB-Net columns teb_* are picked up automatically). DeepTherm's published predictions
             (deeptherm_* columns) are always included.
  --preds    label=path.csv ...  one CSV per extra model, with a `smiles` column (the same strings as in
             --series) and any of  H_kcal, S_cal, Cp300_cal  (prefixes teb_ or pred_ are accepted too).
             Typical controls:
               prior only      preds_prior_v1.csv                 (check/entropy_prior_test.py)
               no prior        output of predict_smiles.py run on the --baseline none checkpoints
               chemprop mean   preds_homologous.csv of check/chemprop_agg_control.py --agg mean
               chemprop sum    preds_homologous.csv of check/chemprop_agg_control.py --agg sum

Usage
  python check/homologous_controls.py --series homologous_predictions.csv \
      --preds "prior only=preds_prior_v1.csv" "no prior=homologous_noprior.csv" \
              "chemprop mean=runs_agg/agg_mean_H_s0/preds_homologous.csv" \
              "chemprop sum=runs_agg/agg_sum_H_s0/preds_homologous.csv" \
      --out_csv homologous_controls.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
PROPS = {
    "H": ("kcal/mol", "kcal"),
    "S": ("cal/(mol K)", "cal"),
    "Cp300": ("cal/(mol K)", "cal"),
}
def _find_col(df: pd.DataFrame, prop: str) -> str | None:
    suf = PROPS[prop][1]
    for c in (f"{prop}_{suf}", f"teb_{prop}_{suf}", f"pred_{prop}_{suf}"):
        if c in df.columns:
            return c
    return None
def load_models(series: pd.DataFrame, preds: list[str]) -> dict[str, pd.DataFrame]:
    """label -> frame aligned with `series` rows, columns H, S, Cp300 (NaN where a model has no value)."""
    models: dict[str, pd.DataFrame] = {}
    def from_cols(df, mapping):
        return pd.DataFrame({p: (df[c].to_numpy(float) if c in df.columns else np.nan) for p, c in mapping.items()},
                            index=series.index)
    if any(c.startswith("teb_") for c in series.columns):
        models["TEB-Net"] = from_cols(series, {p: f"teb_{p}_{PROPS[p][1]}" for p in PROPS})
    for spec in preds:
        if "=" not in spec:
            raise SystemExit(f"--preds expects label=path, got {spec!r}")
        label, path = spec.split("=", 1)
        df = pd.read_csv(path)
        if "smiles" not in df.columns:
            raise SystemExit(f"{path}: no 'smiles' column")
        df = df.drop_duplicates("smiles").set_index(df["smiles"].astype(str).str.strip())
        keys = series["smiles"].astype(str).str.strip()
        missing = (~keys.isin(df.index)).sum()
        if missing:
            print(f"  note: {label}: {missing} of {len(keys)} series SMILES not in {path}")
        cols = {p: _find_col(df, p) for p in PROPS}
        frame = pd.DataFrame(index=series.index)
        for p, c in cols.items():
            frame[p] = keys.map(df[c]).to_numpy(float) if c else np.nan
        models[label.strip()] = frame
    models["DeepTherm (published)"] = from_cols(series, {p: f"deeptherm_{p}_{PROPS[p][1]}" for p in PROPS})
    return models
def series_stats(series: pd.DataFrame, pred: pd.Series, prop: str) -> list[dict]:
    """Per series: line increment, model increment and MAE from the line over the members outside the set,
    and the absolute error against the reference values (ref_* columns, e.g. NIST for n-hexadecane)."""
    suf = PROPS[prop][1]
    data_col, ref_col = f"data_{prop}_{suf}", f"ref_{prop}_{suf}"
    rows = []
    for name, g in series.groupby("series", sort=False):
        g = g.sort_values("n_carbon")
        fit = g[(g.in_deeptherm_set == 1) & (g.n_carbon >= 3) & g[data_col].notna()]
        out = g[g.in_deeptherm_set == 0]
        if len(fit) < 3 or out.empty:
            continue
        slope, icpt = np.polyfit(fit.n_carbon, fit[data_col], 1)
        line = slope * out.n_carbon + icpt
        p = pred.loc[out.index]
        ok = p.notna()
        inc = np.polyfit(out.n_carbon[ok], p[ok], 1)[0] if ok.sum() >= 2 else np.nan
        mae = float(np.abs(p[ok] - line[ok]).mean()) if ok.any() else np.nan
        ref = out[ref_col] if ref_col in out.columns else pd.Series(np.nan, index=out.index)
        r_ok = ok & ref.notna()
        rows.append({"series": name, "prop": prop, "n_range": f"C{out.n_carbon.min()}–C{out.n_carbon.max()}",
                     "line_increment": slope, "increment": inc, "mae_line": mae,
                     "n_ref": int(r_ok.sum()),
                     "mae_ref": float(np.abs(p[r_ok] - ref[r_ok]).mean()) if r_ok.any() else np.nan})
    return rows
def report(series: pd.DataFrame, models: dict[str, pd.DataFrame], props=("H", "S", "Cp300")) -> pd.DataFrame:
    long = []
    for prop in props:
        unit = PROPS[prop][0]
        per_model = {}
        for label, frame in models.items():
            if frame[prop].notna().any():
                per_model[label] = series_stats(series, frame[prop], prop)
                for r in per_model[label]:
                    long.append({"model": label, **r})
        if not per_model:
            continue
        labels = list(per_model)
        print(f"\n### {prop} ({unit}): increment per CH2 / MAE from the line, members outside the DeepTherm set\n")
        print("| Series | Outside | Line | " + " | ".join(labels) + " |")
        print("|---|---|---|" + "---|" * len(labels))
        first = per_model[labels[0]]
        for i, r0 in enumerate(first):
            cells = []
            for lab in labels:
                r = per_model[lab][i]
                cells.append(f"{r['increment']:+.2f} / {r['mae_line']:.2f}")
            print(f"| {r0['series']} | {r0['n_range']} | {r0['line_increment']:+.2f} | " + " | ".join(cells) + " |")
        refs = [lab for lab in labels if any(r["n_ref"] for r in per_model[lab])]
        if refs:
            print("| vs reference (NIST) | | | " + " | ".join(
                (f"{np.nanmean([r['mae_ref'] for r in per_model[lab] if r['n_ref']]):.2f}" if lab in refs else "–")
                for lab in labels) + " |")
    df = pd.DataFrame(long)
    if not df.empty:
        print("\n### Summary per model (members outside the set)\n")
        print("| Model | Property | Increment range | Line range | Mean MAE from the line |")
        print("|---|---|---|---|---|")
        for (lab, prop), g in df.groupby(["model", "prop"], sort=False):
            print(f"| {lab} | {prop} | {g.increment.min():+.2f} to {g.increment.max():+.2f} | "
                  f"{g.line_increment.min():+.2f} to {g.line_increment.max():+.2f} | {g.mae_line.mean():.2f} |")
    return df
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", required=True)
    ap.add_argument("--preds", nargs="*", default=[])
    ap.add_argument("--props", nargs="+", default=["H", "S", "Cp300"], choices=list(PROPS))
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()
    series = pd.read_csv(args.series)
    need = {"series", "n_carbon", "smiles", "in_deeptherm_set"}
    if need - set(series.columns):
        raise SystemExit(f"{args.series}: missing columns {sorted(need - set(series.columns))}")
    models = load_models(series, args.preds)
    print("models:", ", ".join(models))
    df = report(series, models, args.props)
    if args.out_csv:
        df.to_csv(args.out_csv, index=False, float_format="%.3f")
        print(f"\nwrote {Path(args.out_csv)}")
if __name__ == "__main__":
    main()

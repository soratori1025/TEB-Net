"""Summarise TEB-Net runs made with scripts/train.py under the DeepTherm protocol.

For every (model tag, split) it reports, in DeepTherm units (ΔH kcal/mol; S, Cp cal/(mol·K)):
  * single-model test MAE of each member and mean ± std over members / init seeds
  * the weighted ensemble of the members (w_i ∝ 1 / validation MAE averaged over the 9 targets,
    as in DeepTherm §2.3) — members must share the test set (use --member, not different --split_seed)
  * ensemble breakdown: closed-shell vs radicals, cyclic vs acyclic, and test MAE vs number of heavy
    atoms (as in DeepTherm Fig. 6g)
and, over split seeds (member 0 of each), mean ± std of the single-model test MAE.

Usage:  python scripts/deeptherm_eval.py --runs benchmark_results_deeptherm [--csv summary.csv]
        # supplementary evaluation without confirmed label errors (same rows for every model):
        python scripts/deeptherm_eval.py --runs ... --exclude_file excluded_rows.txt
Needs only numpy.
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
J_TO_KCAL = 1.0 / 4184.0
J_TO_CAL = 1.0 / 4.184
def read_exclude(rows: list[int] | None, path: str | None) -> set[int]:
    """Dataset rows (0-based, = CSV data rows = results.json indices) to drop from every TEST set."""
    out = set(int(r) for r in (rows or []))
    if path:
        for line in Path(path).read_text().splitlines():
            line = line.split("#", 1)[0].replace(",", " ")
            out.update(int(tok) for tok in line.split())
    return out
def apply_exclude(runs: list[dict], exclude: set[int]) -> None:
    """Removes the excluded rows from the TEST predictions of every run, in place. Validation sets
    (and hence checkpoint choice and ensemble weights) are left as they were during training."""
    if not exclude:
        return
    ex = np.array(sorted(exclude))
    reported = set()
    for r in runs:
        idx = np.asarray(r["test_idx"])
        keep = ~np.isin(idx, ex)
        removed = tuple(int(i) for i in idx[~keep])
        r["p"] = {k: v[keep] for k, v in r["p"].items()}
        r["test_idx"] = tuple(int(i) for i in idx[keep])
        if r["smiles"] is not None:
            r["smiles"] = [s for s, k in zip(r["smiles"], keep) if k]
        r["n_excluded"] = len(removed)
        sp = r["res"]["split"]
        key = (sp["type"], sp["seed"], sp.get("test_smarts"), removed)
        if key not in reported:
            reported.add(key)
            print(f"  [exclude] {sp['type']}{sp['seed']}"
                  + (f" [{sp['test_smarts']}]" if sp.get("test_smarts") else "")
                  + f": {len(removed)} of {len(idx)} test molecules removed {list(removed) if removed else ''}")
def load_runs(roots: list[str]) -> list[dict]:
    runs = []
    for root in roots:
        for rp in sorted(Path(root).rglob("results.json")):
            try:
                res = json.loads(rp.read_text())
            except Exception:
                continue
            if res.get("protocol") != "deeptherm":
                continue
            npz_path = rp.parent / "predictions.npz"
            if not npz_path.exists():
                continue
            z = np.load(npz_path, allow_pickle=True)
            p = {k: z[f"test_{k}"] for k in ("h_pred", "s_pred", "cp_pred", "h_true", "s_true", "cp_true", "x")}
            runs.append({
                "path": str(rp.parent), "res": res, "p": p,
                "fn": [str(n) for n in z["feature_names"]],
                "t_grid": np.asarray(res.get("t_grid", z["t_grid"] if "t_grid" in z else []), dtype=float),
                "test_idx": tuple(res["test_indices"]),
                "smiles": [str(x) for x in z["test_smiles"]] if "test_smiles" in z else None,
                "pv": ({k: z[f"val_{k}"] for k in ("h_pred", "s_pred", "cp_pred", "h_true", "s_true", "cp_true", "x")}
                       if "val_x" in z else None),
                "val_smiles": [str(x) for x in z["val_smiles"]] if "val_smiles" in z else None,
            })
    return runs
def _mae(pred, true, mask=None):
    m = np.isfinite(pred) & np.isfinite(true)
    if mask is not None:
        m = m & mask
    return float(np.mean(np.abs(pred[m] - true[m]))) if m.any() else float("nan")
def metrics(p: dict, t_grid: np.ndarray, mask=None) -> dict:
    out = {"H": _mae(p["h_pred"] * J_TO_KCAL, p["h_true"] * J_TO_KCAL, mask),
           "S": _mae(p["s_pred"] * J_TO_CAL, p["s_true"] * J_TO_CAL, mask)}
    cps = []
    for j, T in enumerate(t_grid):
        v = _mae(p["cp_pred"][:, j] * J_TO_CAL, p["cp_true"][:, j] * J_TO_CAL, mask)
        out[f"Cp{T:.0f}"] = v
        cps.append(v)
    out["Cp_mean"] = float(np.nanmean(cps))
    out["mean9"] = float(np.nanmean([out["H"], out["S"], *cps]))
    out["n"] = int(np.sum(mask)) if mask is not None else len(p["h_true"])
    return out
def fmt_row(label: str, m: dict, width: int = 34) -> str:
    return (f"  {label:<{width}s} {m['n']:>5d} {m['H']:>8.3f} {m['S']:>8.3f} {m.get('Cp300', float('nan')):>8.3f} "
            f"{m['Cp_mean']:>8.3f} {m['mean9']:>8.3f}")
HEADER = f"  {'':<34s} {'n':>5s} {'ΔH298':>8s} {'S298':>8s} {'Cp300':>8s} {'Cp mean':>8s} {'mean-9':>8s}"
def ensemble(members: list[dict]) -> tuple[dict, np.ndarray]:
    val = np.array([m["res"].get("val_deeptherm_mae", np.nan) for m in members], dtype=float)
    w = 1.0 / np.maximum(val, 1e-12)
    w = w / w.sum() if np.isfinite(w).all() else np.full(len(members), 1.0 / len(members))
    p0 = members[0]["p"]
    ens = {k: p0[k] for k in ("h_true", "s_true", "cp_true", "x")}
    for k in ("h_pred", "s_pred", "cp_pred"):
        ens[k] = np.tensordot(w, np.stack([m["p"][k] for m in members]), axes=(0, 0))
    return ens, w
def worst(runs: list[dict], tag: str, n: int, prop: str, which: str = "val") -> None:
    sel = [r for r in runs if r["res"]["tag"] == tag and r["res"]["member"] == 0]
    if not sel:
        print(f"\n[worst] tag {tag!r} not found")
        return
    for r in sorted(sel, key=lambda r: (r["res"]["split"]["type"], r["res"]["split"]["seed"])):
        fn, t_grid = r["fn"], r["t_grid"]
        if which == "val":
            if r["pv"] is None:
                print(f"\n[worst] {r['path']}: no validation predictions saved")
                continue
            p, smiles = r["pv"], r["val_smiles"]
        else:
            p, smiles = r["p"], r["smiles"]
        col = {k: i for i, k in enumerate(fn)}
        if prop == "H":
            pred, true, unit = p["h_pred"] * J_TO_KCAL, p["h_true"] * J_TO_KCAL, "kcal/mol"
        elif prop == "S":
            pred, true, unit = p["s_pred"] * J_TO_CAL, p["s_true"] * J_TO_CAL, "cal/(mol K)"
        else:
            j = int(np.argmin(np.abs(t_grid - 300.0)))
            pred, true, unit = p["cp_pred"][:, j] * J_TO_CAL, p["cp_true"][:, j] * J_TO_CAL, "cal/(mol K)"
        err = np.abs(pred - true)
        ok = np.isfinite(err)
        e = err[ok]
        sp = r["res"]["split"]
        print("\n" + "-" * 92)
        print(f" worst {prop} errors on the {which.upper()} set | {tag} | {sp['type']}{sp['seed']} | {unit}")
        print(f"  MAE {e.mean():.3f} | median {np.median(e):.3f} | 90th pct {np.percentile(e, 90):.3f} | "
              f"errors > 5: {(e > 5).sum()} ({100 * (e > 5).mean():.1f}%) carrying {100 * e[e > 5].sum() / e.sum():.0f}% of the total error")
        order = np.argsort(-np.where(ok, err, -1))[:n]
        print(f"  {'SMILES':<40s} {'true':>9s} {'pred':>9s} {'|err|':>7s} {'heavy':>5s} {'rad':>4s} {'rings':>5s}")
        for i in order:
            smi = smiles[i] if smiles else "?"
            print(f"  {smi[:40]:<40s} {true[i]:>9.2f} {pred[i]:>9.2f} {err[i]:>7.2f} "
                  f"{int(p['x'][i, col['n_heavy']]):>5d} {int(p['x'][i, col['n_rad_e']] > 0):>4d} "
                  f"{int(p['x'][i, col['n_rings']]):>5d}")
def _weighted(members: list[dict], val_key) -> tuple[dict, np.ndarray]:
    """Weighted average of members' test predictions, w ∝ 1/val_key(member)."""
    v = np.array([val_key(m) for m in members], dtype=float)
    w = 1.0 / np.maximum(v, 1e-12)
    w = w / w.sum() if np.isfinite(w).all() else np.full(len(members), 1.0 / len(members))
    p0 = members[0]["p"]
    out = {k: p0[k] for k in ("h_true", "s_true", "cp_true", "x")}
    for k in ("h_pred", "s_pred", "cp_pred"):
        out[k] = np.tensordot(w, np.stack([m["p"][k] for m in members]), axes=(0, 0))
    return out, w
def _val_h(m):
    return m["res"]["val"]["H298_kcal"]["mae"]
def _val_scp(m):
    v = m["res"]["val"]
    n_t = len(m["t_grid"])
    return (v["S298_cal"]["mae"] + n_t * v["Cp_mean_MAE_cal"]) / (1 + n_t)
def combine(runs: list[dict], h_tag: str, scp_tag: str, rows: list) -> None:
    """Decoupled model: ΔH from H_TAG runs, S and Cp from SCP_TAG runs.
    Single = member 0 (lowest seed) of each. Ensemble = all runs of each tag that share the test set
    (members and/or init seeds), weighted by 1/val-MAE of the property that model was trained for.
    Averaging keeps the NASA-7 output exact: H298, S298 and the Cp polynomial enter the analytic layer
    linearly, so a convex combination of consistent members is itself consistent."""
    by_tag: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    for r in sorted(runs, key=lambda r: (r["res"]["member"], r["res"]["seed"])):
        sp = r["res"]["split"]
        by_tag[r["res"]["tag"]][(sp["type"], sp["seed"], sp.get("test_smarts"))].append(r)
    for t in (h_tag, scp_tag):
        if t not in by_tag:
            print(f"\n[combine] tag {t!r} not found. Available tags:\n  " + "\n  ".join(sorted(by_tag)))
            return
    keys = sorted(set(by_tag[h_tag]) & set(by_tag[scp_tag]), key=str)
    print("\n" + "#" * 92)
    print(f" DECOUPLED  ΔH <- {h_tag}\n            S, Cp <- {scp_tag}")
    print("#" * 92)
    print(HEADER)
    singles, ensembles = [], []
    for key in keys:
        hs, ss = by_tag[h_tag][key], by_tag[scp_tag][key]
        test_sets = {r["test_idx"] for r in hs + ss}
        if len(test_sets) != 1:
            print(f"  {key}: runs have different test sets, skipped")
            continue
        name = f"{key[0]}{key[1]}" + (f" [{key[2]}]" if key[2] else "")
        p = dict(ss[0]["p"])
        p["h_pred"] = hs[0]["p"]["h_pred"]
        mm = metrics(p, ss[0]["t_grid"])
        singles.append(mm)
        print(fmt_row(f"{name}  single", mm))
        rows.append({"tag": f"DECOUPLED[{h_tag}|{scp_tag}]", "split": name, "kind": "decoupled-single",
                     "member": 0, "seed": -1, **mm})
        if len(hs) > 1 or len(ss) > 1:
            eh, _ = _weighted(hs, _val_h)
            es, _ = _weighted(ss, _val_scp)
            pe = dict(es)
            pe["h_pred"] = eh["h_pred"]
            me = metrics(pe, ss[0]["t_grid"])
            ensembles.append(me)
            print(fmt_row(f"{name}  ENSEMBLE ({len(hs)} ΔH + {len(ss)} S/Cp)", me))
            rows.append({"tag": f"DECOUPLED[{h_tag}|{scp_tag}]", "split": name, "kind": "decoupled-ensemble",
                         "member": -1, "seed": -1, **me})
    ks = ["H", "S", "Cp300", "Cp_mean", "mean9"]
    for lab, ms in (("single", singles), ("ensemble", ensembles)):
        if len(ms) > 1:
            print(f"  {lab + ' mean':<34s} {'':>5s} " + " ".join(f"{np.nanmean([m[k] for m in ms]):>8.3f}" for k in ks))
            print(f"  {lab + ' std':<34s} {'':>5s} " + " ".join(f"{np.nanstd([m[k] for m in ms], ddof=1):>8.3f}" for k in ks))
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["benchmark_results_deeptherm"])
    ap.add_argument("--csv", default=None, help="write a flat summary CSV here")
    ap.add_argument("--worst_tag", default=None, help="list the worst test molecules of this tag")
    ap.add_argument("--worst", type=int, default=25)
    ap.add_argument("--worst_prop", choices=["H", "S", "Cp300"], default="H")
    ap.add_argument("--worst_set", choices=["val", "test"], default="val",
                    help="error analysis on the VALIDATION set by default, so that feature design is never "
                         "driven by test molecules")
    ap.add_argument("--combine", nargs=2, metavar=("H_TAG", "SCP_TAG"), default=None,
                    help="decoupled model: ΔH from runs with tag H_TAG, S and Cp from runs with tag SCP_TAG "
                         "(same splits). The analytic NASA-7 layer is still consistent: a6/a7 follow from "
                         "H298, S298 and the Cp polynomial, whichever model produced them.")
    ap.add_argument("--exclude_rows", type=int, nargs="*", default=None,
                    help="dataset rows (0-based, as in results.json / the BDE audit) removed from every TEST "
                         "set before scoring, e.g. confirmed label errors. Validation sets are not touched.")
    ap.add_argument("--exclude_file", default=None,
                    help="text file with rows to exclude (whitespace/comma separated, '#' comments)")
    args = ap.parse_args()
    runs = load_runs(args.runs)
    if not runs:
        raise SystemExit(f"no DeepTherm-protocol runs under {args.runs}")
    exclude = read_exclude(args.exclude_rows, args.exclude_file)
    if exclude:
        print(f"\nCLEANED TEST SETS: {len(exclude)} dataset rows excluded from scoring")
        apply_exclude(runs, exclude)
    rows = []
    groups = defaultdict(list)
    for r in runs:
        sp = r["res"]["split"]
        groups[(r["res"]["tag"], sp["type"], sp["seed"], sp.get("test_smarts"))].append(r)
    for (tag, stype, sseed, smarts), members in sorted(groups.items(), key=lambda kv: str(kv[0])):
        t_grid = members[0]["t_grid"]
        name = f"{stype}{sseed}" + (f" [{smarts}]" if smarts else "")
        print("\n" + "=" * 92)
        print(f" {tag} | split {name} | {len(members)} run(s)"
              + ("" if members[0]["res"]["split"].get("same_order_as_deeptherm_SI")
                 else "   (row order differs from DeepTherm SI)"))
        print("=" * 92)
        print(HEADER)
        per = []
        for m in sorted(members, key=lambda m: (m["res"]["member"], m["res"]["seed"])):
            mm = metrics(m["p"], t_grid)
            per.append(mm)
            lab = f"member {m['res']['member']} seed {m['res']['seed']} (val {m['res'].get('val_deeptherm_mae', float('nan')):.3f})"
            print(fmt_row(lab, mm))
            tr = m["res"].get("train", {})
            if tr:
                print(f"  {'   train (same model)':<34s} {'':>5s} {tr['H298_kcal']['mae']:>8.3f} {tr['S298_cal']['mae']:>8.3f} "
                      f"{tr.get('Cp300_cal', {}).get('mae', float('nan')):>8.3f} {tr['Cp_mean_MAE_cal']:>8.3f} {tr['mean9_MAE']:>8.3f}")
            rows.append({"tag": tag, "split": name, "kind": "single", "member": m["res"]["member"],
                         "seed": m["res"]["seed"], **mm})
        if len(per) > 1:
            keys = ["H", "S", "Cp300", "Cp_mean", "mean9"]
            mean = {k: np.nanmean([q[k] for q in per]) for k in keys}
            std = {k: np.nanstd([q[k] for q in per], ddof=1) for k in keys}
            print(f"  {'single-model mean':<34s} {'':>5s} " + " ".join(f"{mean[k]:>8.3f}" for k in keys))
            print(f"  {'single-model std':<34s} {'':>5s} " + " ".join(f"{std[k]:>8.3f}" for k in keys))
        same_test = len({m["test_idx"] for m in members}) == 1
        if len(members) > 1 and same_test:
            ens, w = ensemble(members)
            em = metrics(ens, t_grid)
            print(fmt_row(f"WEIGHTED ENSEMBLE ({len(members)})", em))
            rows.append({"tag": tag, "split": name, "kind": "ensemble", "member": -1, "seed": -1, **em})
            fn = members[0]["fn"]
            col = {n: i for i, n in enumerate(fn)}
            x = ens["x"]
            rad = x[:, col["n_rad_e"]] > 0
            nring = x[:, col["n_rings"]]
            print("\n  ensemble by class")
            print(HEADER)
            for lab, mask in [("closed-shell", ~rad), ("radicals", rad), ("acyclic", nring == 0),
                              ("cyclic", nring > 0), ("polycyclic", nring >= 2)]:
                if mask.any():
                    print(fmt_row(lab, metrics(ens, t_grid, mask)))
            heavy = x[:, col["n_heavy"]].astype(int)
            print("\n  ensemble by number of heavy atoms (cf. DeepTherm Fig. 6g)")
            print(HEADER)
            for h in sorted(set(heavy.tolist())):
                mask = heavy == h
                if mask.sum() >= 3:
                    print(fmt_row(f"heavy atoms = {h}", metrics(ens, t_grid, mask)))
        elif len(members) > 1:
            print("  (members have different test sets -> no ensemble; use --member for ensembles)")
    by_split = defaultdict(list)
    for r in runs:
        if r["res"]["member"] == 0:
            sp = r["res"]["split"]
            by_split[(r["res"]["tag"], sp["type"], sp.get("test_smarts"))].append(r)
    for (tag, stype, smarts), rs in sorted(by_split.items(), key=lambda kv: str(kv[0])):
        seeds = sorted({r["res"]["split"]["seed"] for r in rs})
        if len(seeds) < 2:
            continue
        ms = [metrics(r["p"], r["t_grid"]) for r in rs]
        keys = ["H", "S", "Cp300", "Cp_mean", "mean9"]
        print("\n" + "-" * 92)
        print(f" {tag} | {stype}{' [' + smarts + ']' if smarts else ''} | single models over split seeds {seeds}")
        print(f"  {'':<34s} {'':>5s} {'ΔH298':>8s} {'S298':>8s} {'Cp300':>8s} {'Cp mean':>8s} {'mean-9':>8s}")
        print(f"  {'mean':<34s} {'':>5s} " + " ".join(f"{np.nanmean([m[k] for m in ms]):>8.3f}" for k in keys))
        print(f"  {'std':<34s} {'':>5s} " + " ".join(f"{np.nanstd([m[k] for m in ms], ddof=1):>8.3f}" for k in keys))
    if args.combine:
        combine(runs, *args.combine, rows)
    if args.worst_tag:
        worst(runs, args.worst_tag, args.worst, args.worst_prop, args.worst_set)
    if args.csv:
        import csv
        keys = sorted({k for r in rows for k in r})
        with open(args.csv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            wr.writerows(rows)
        print(f"\nwrote {args.csv}")
if __name__ == "__main__":
    main()

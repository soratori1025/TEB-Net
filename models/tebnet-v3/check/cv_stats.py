"""Statistics for the nested-CV runs of scripts/train_cv.py (no training, numpy only; RDKit optional).

For one tag, or a decoupled pair (ΔH from H_TAG, S/Cp from SCP_TAG), it prints in kJ/mol and J/(mol·K):
  * MAE and RMSE over the whole dataset, with a bootstrap 95 % CI of the MAE (molecules resampled)
  * MAE per outer fold (mean ± std over the 10 folds)
  * optionally, where a published reference MAE (e.g. Ureel's transfer learning) falls: inside the CI or not,
    and the bootstrap fraction of resamples in which TEB-Net's MAE is below it
  * duplicate molecules (same InChIKey / canonical SMILES) and whether copies sit in different outer folds,
    which would let one copy be trained on while the other is tested
  * the N largest ΔH errors

Usage
  python scripts/cv_stats.py --run_dir benchmark_results_cv/ab_initio_database_rad \
      --tags <H_TAG> <SCP_TAG> --ref 2.17 2.79 1.65 --worst 10 --drop_dups
  (--ref = published ΔH [kJ/mol], S298 and Cp300 [J/(mol·K)] MAEs to compare with; optional)
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_tag(run_dir: Path):
    files = sorted(run_dir.glob("outer*.npz"))
    if not files:
        raise SystemExit(f"no outer*.npz in {run_dir}")
    parts = [np.load(f, allow_pickle=True) for f in files]
    fold = np.concatenate([np.full(len(z["test_idx"]), int(f.stem.replace("outer", ""))) for f, z in zip(files, parts)])
    idx = np.concatenate([z["test_idx"] for z in parts])
    order = np.argsort(idx)
    p = {k: np.concatenate([z[k] for z in parts])[order]
         for k in ("h_pred", "s_pred", "cp_pred", "h_true", "s_true", "cp_true")}
    p["smiles"] = np.concatenate([z["smiles"] for z in parts])[order]
    p["fold"] = fold[order]
    meta = json.loads(str(parts[0]["meta"]))
    if len(files) != meta["n_outer"]:
        print(f"  WARNING: {run_dir.name}: only {len(files)}/{meta['n_outer']} outer folds present")
    return p, idx[order], meta


def errors(p: dict, t_grid: np.ndarray) -> dict[str, np.ndarray]:
    j300 = int(np.argmin(np.abs(t_grid - 300.0)))
    return {"ΔH298 [kJ/mol]": (p["h_pred"] - p["h_true"]) / 1000.0,
            "S298 [J/(mol K)]": p["s_pred"] - p["s_true"],
            "Cp300 [J/(mol K)]": p["cp_pred"][:, j300] - p["cp_true"][:, j300]}


def boot_mae(err: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    a = np.abs(err[np.isfinite(err)])
    return a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(1)


def duplicates(smiles: np.ndarray, fold: np.ndarray, h_true: np.ndarray) -> np.ndarray:
    """Prints duplicate groups; returns a boolean mask of rows in groups that span several outer folds
    (one copy trained on while the other is tested). The ΔH spread inside a group tells exact duplicates
    (spread ~0) from distinct species that share an InChIKey (e.g. E/Z isomers of a vinyl radical)."""
    try:
        from rdkit import Chem

        def key(s):
            m = Chem.MolFromSmiles(str(s))
            if m is None:
                return str(s)
            try:
                return Chem.MolToInchiKey(m) or Chem.MolToSmiles(m)
            except Exception:
                return Chem.MolToSmiles(m)
        how = "InChIKey"
    except ImportError:
        def key(s):
            return str(s)
        how = "SMILES string (RDKit not installed)"
    groups = defaultdict(list)
    for i, s in enumerate(smiles):
        groups[key(s)].append(i)
    dup = [g for g in groups.values() if len(g) > 1]
    cross = [g for g in dup if len({int(fold[i]) for i in g}) > 1]
    print(f"\n  duplicates by {how}: {len(dup)} groups ({sum(len(g) for g in dup)} rows); "
          f"{len(cross)} groups split across outer folds")
    for g in cross:
        spread = (h_true[g].max() - h_true[g].min()) / 1000.0
        print(f"    ΔH spread {spread:6.2f} kJ/mol | " + " | ".join(f"{smiles[i]} (fold {int(fold[i])})" for i in g))
    mask = np.zeros(len(smiles), dtype=bool)
    for g in cross:
        mask[g] = True
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="benchmark_results_cv/<dataset stem>")
    ap.add_argument("--tags", nargs="+", required=True, help="one tag, or H_TAG SCP_TAG (decoupled)")
    ap.add_argument("--ref", type=float, nargs=3, default=None, metavar=("H_KJ", "S_J", "CP300_J"),
                    help="published MAEs to compare with (e.g. Ureel transfer learning)")
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--worst", type=int, default=10)
    ap.add_argument("--drop_dups", action="store_true",
                    help="also report the metrics without molecules whose duplicate is in another outer fold")
    args = ap.parse_args()

    root = Path(args.run_dir)
    p, idx, meta = load_tag(root / args.tags[0])
    if len(args.tags) == 2:
        ps, idx_s, _ = load_tag(root / args.tags[1])
        if not np.array_equal(idx, idx_s):
            raise SystemExit("the two tags do not cover the same molecules (different --cv_seed?)")
        for k in ("s_pred", "cp_pred"):
            p[k] = ps[k]
    t_grid = np.asarray(meta["t_grid"])
    rng = np.random.default_rng(args.seed)

    print(f"\n{root.name} | {' + '.join(args.tags)} | n = {len(idx)}")
    dup_mask = duplicates(p["smiles"], p["fold"], p["h_true"])

    def table(keep: np.ndarray, title: str) -> None:
        print(f"\n  {title} (n = {int(keep.sum())})")
        if keep.sum() < 2:
            print("  too few molecules left to score")
            return
        print(f"  {'':<20s} {'MAE':>7s} {'95 % CI':>17s} {'RMSE':>7s} {'per fold':>15s}"
              + (f" {'ref':>6s} {'ref in CI':>9s} {'P(MAE<ref)':>10s}" if args.ref else ""))
        for (name, e), ref in zip(errors(p, t_grid).items(), args.ref or [None] * 3):
            ok = np.isfinite(e) & keep
            mae, rmse = np.abs(e[ok]).mean(), np.sqrt(np.square(e[ok]).mean())
            b = boot_mae(e[ok], args.n_boot, rng)
            lo, hi = np.percentile(b, [2.5, 97.5])
            per = [np.abs(e[(p["fold"] == f) & ok]).mean() for f in np.unique(p["fold"])]
            line = (f"  {name:<20s} {mae:>7.2f} [{lo:>6.2f}, {hi:>6.2f}] {rmse:>7.2f} "
                    f"{np.mean(per):>7.2f} ± {np.std(per, ddof=1):>4.2f}")
            if ref is not None:
                line += f" {ref:>6.2f} {('yes' if lo <= ref <= hi else 'no'):>9s} {np.mean(b < ref):>10.2f}"
            print(line)

    table(np.ones(len(idx), dtype=bool), "all molecules")
    if args.drop_dups and dup_mask.any():
        # sensitivity check: score only molecules with no copy in another outer fold (no retraining needed;
        # a duplicate can only help the prediction of its own copy)
        table(~dup_mask, "without molecules whose duplicate sits in another outer fold")

    if args.worst:
        e = np.abs(errors(p, t_grid)["ΔH298 [kJ/mol]"])
        order = np.argsort(-np.where(np.isfinite(e), e, -1))[: args.worst]
        print(f"\n  {args.worst} largest ΔH298 errors [kJ/mol]")
        print(f"  {'SMILES':<40s} {'true':>9s} {'pred':>9s} {'|err|':>7s} {'fold':>4s}")
        for i in order:
            print(f"  {str(p['smiles'][i])[:40]:<40s} {p['h_true'][i] / 1000:>9.2f} {p['h_pred'][i] / 1000:>9.2f} "
                  f"{e[i]:>7.2f} {int(p['fold'][i]):>4d}")


if __name__ == "__main__":
    main()
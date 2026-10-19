"""Per-run training diagnostics for TEB-Net runs (DeepTherm protocol).

For every results.json under --runs this prints the epochs trained, the epoch of the kept checkpoint,
whether early stopping fired, the train and test MAE, and the σ, multiplicity and chirality coefficients
of the frozen S baseline, in units of R. These coefficients are shown for tracking only: they are not
identifiable as −R ln σ / R ln(2S+1) because of collinearity with n_methyl, n_rotors and the ring
counts (observed: log_sigma ≈ −0.3 R, ln_mult ≈ +4 R with the corrected σ).

With the cosine LR schedule, a kept checkpoint far before max_epochs, or a train MAE well above the
other runs of the same tag, means the run was cut short (random1 S/Cp run in v15.3: train S 1.15 vs
0.50 on random0).

--rerun FILE writes commands that re-run the early-stopped runs with the current train.py defaults
(no early stopping) into the same directories (--overwrite). Runs that reached max_epochs are
unaffected by the change of default, so only these need to be redone. Run the commands only with the
SAME feature version as the original runs (column 'feat').

  python scripts/run_info.py --runs benchmark_results_v153
  python scripts/run_info.py --runs benchmark_results_v153 --tag mon-scp --rerun rerun_early.sh
"""
from __future__ import annotations

import argparse
import json
import shlex
from collections import defaultdict
from pathlib import Path

import numpy as np


def ckpt_best_epoch(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        import torch
        return int(torch.load(path, map_location="cpu", weights_only=False)["epoch"]) + 1
    except Exception:
        return None


def feature_version(r: dict) -> str:
    if r.get("features_version"):
        return str(r["features_version"])
    coefs = r.get("baseline_coefficients", {}) or {}
    if "H:rad_ethynyl" in coefs:
        return "v15.3*"      # inferred from the radical types in the baseline report
    if "H:rad_vinyl" in coefs:
        return "v15.2*"
    return "?"


def monitored_train(r: dict) -> float:
    """Train MAE of the quantity used for model selection (same definition as the val monitor)."""
    tr = r["train"]
    h, s, cpm = tr["H298_kcal"]["mae"], tr["S298_cal"]["mae"], tr["Cp_mean_MAE_cal"]
    mon = (r.get("args") or {}).get("monitor", "deeptherm")
    return {"h": h, "s": s, "scp": (s + 7.0 * cpm) / 8.0,
            "cp300": tr.get("Cp300_cal", {}).get("mae", cpm)}.get(mon, tr["mean9_MAE"])


def rerun_cmd(a: dict) -> str:
    parts = ["python", "scripts/train.py"]
    for k, v in a.items():
        if k in ("patience", "overwrite") or v is None or v is False:
            continue
        if v is True:
            parts.append(f"--{k}")
        elif isinstance(v, (list, tuple)):
            parts += [f"--{k}", *(f"{x:g}" if isinstance(x, float) else str(x) for x in v)]
        else:
            parts += [f"--{k}", str(v)]
    parts.append("--overwrite")
    return " ".join(shlex.quote(p) for p in parts)


def fmt(x, w=6, d=3):
    return f"{x:>{w}.{d}f}" if isinstance(x, (int, float)) and x is not None and np.isfinite(x) else f"{'-':>{w}s}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="output directory of train.py (e.g. benchmark_results_v153)")
    ap.add_argument("--tag", default="", help="only tags containing this substring")
    ap.add_argument("--rerun", default=None, help="write re-run commands for early-stopped runs to this file")
    ap.add_argument("--underfit_ratio", type=float, default=1.5,
                    help="flag a run whose monitored train MAE exceeds this × the median of its tag")
    args = ap.parse_args()

    groups: dict[str, list] = defaultdict(list)
    for f in sorted(Path(args.runs).rglob("results.json")):
        try:
            r = json.loads(f.read_text())
        except Exception as e:
            print(f"skip {f}: {e}")
            continue
        tag = r.get("tag", f.parents[2].name)
        if args.tag in tag:
            groups[tag].append((f, r))

    reruns, skipped = [], []
    for tag, runs in sorted(groups.items()):
        med = float(np.median([monitored_train(r) for _, r in runs]))
        print("=" * 124)
        print(f" {tag}   ({len(runs)} runs | median monitored train MAE {med:.3f})")
        print("=" * 124)
        print(f"  {'split':<24s} {'m':>2s} {'seed':>4s} {'feat':>7s} {'ep':>4s} {'best':>4s} {'ES':>3s} {'pat':>4s} |"
              f" {'trainH':>6s} {'S':>6s} {'CpM':>6s} | {'testH':>6s} {'S':>6s} {'CpM':>6s} |"
              f" {'σ[R]':>6s} {'mult':>6s} {'chir':>6s}  flags")
        for f, r in sorted(runs, key=lambda x: str(x[0])):
            a = r.get("args") or {}
            split = f.parents[1].name
            max_ep = int(a.get("max_epochs", 100))
            ep = r.get("epochs_trained")
            best = r.get("best_epoch")
            if best is None:
                ck = Path(r.get("checkpoint") or "")
                best = ckpt_best_epoch(ck if ck.name else f.parent / "best.ckpt")
                if best is None:
                    best = ckpt_best_epoch(f.parent / "best.ckpt")
            es = ep is not None and ep < max_ep
            tr, te = r["train"], r["test"]
            c = r.get("baseline_coefficients", {}) or {}
            flags = []
            if es:
                flags.append("EARLY-STOPPED")
            if best is not None and best < 0.5 * max_ep:
                flags.append(f"ckpt@{best}<{max_ep // 2}")
            if monitored_train(r) > args.underfit_ratio * med:
                flags.append(f"train {monitored_train(r) / med:.1f}×median")
            print(f"  {split:<24s} {r.get('member', 0):>2} {r.get('seed', 0):>4} {feature_version(r):>7s} "
                  f"{ep if ep is not None else '-':>4} {best if best is not None else '-':>4} "
                  f"{'yes' if es else 'no':>3s} {str(a.get('patience', '-')):>4s} |"
                  f" {fmt(tr['H298_kcal']['mae'])} {fmt(tr['S298_cal']['mae'])} {fmt(tr['Cp_mean_MAE_cal'])} |"
                  f" {fmt(te['H298_kcal']['mae'])} {fmt(te['S298_cal']['mae'])} {fmt(te['Cp_mean_MAE_cal'])} |"
                  f" {fmt(c.get('S:log_sigma [R]'), 6, 2)} {fmt(c.get('S:ln_mult [R]'), 6, 2)}"
                  f" {fmt(c.get('S:is_chiral [R]'), 6, 2)}  {' '.join(flags)}")
            if es:
                if "_pat" in tag:
                    skipped.append(f"{tag}/{split}: patience was set explicitly (tag has _pat) — not re-run")
                else:
                    reruns.append((feature_version(r), f"{tag}/{split}", rerun_cmd(a)))

    print("\nS-baseline coefficients (units of R) are NOT physical constants: log_sigma is collinear with n_methyl "
          "(each CH3 rotor adds ln 3),\n  n_rotors and the ring / aromatic counts, and ln_mult is the only radical "
          "term in the S baseline (no n_rad_e), so it\n  absorbs the whole radical-vs-closed-shell entropy difference. "
          "Validate σ with scripts/check_sigma.py, not with these numbers.")
    if reruns:
        print(f"\n{len(reruns)} early-stopped run(s):")
        for fv, name, _ in reruns:
            print(f"  [{fv}] {name}")
        if args.rerun:
            with open(args.rerun, "w") as fh:
                fh.write("#!/usr/bin/env bash\n# Re-run early-stopped runs with the current train.py defaults "
                         "(no early stopping).\n# Check that the feature version in brackets matches the code "
                         "you have installed.\nset -e\n")
                for fv, name, cmd in reruns:
                    fh.write(f"# [{fv}] {name}\n{cmd}\n")
            print(f"wrote {args.rerun}")
    else:
        print("\nno early-stopped runs")
    for s in skipped:
        print("  " + s)


if __name__ == "__main__":
    main()
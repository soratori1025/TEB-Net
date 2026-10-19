"""Small numbers the manuscript still needs (no training, CPU only).

  rings     ring counts by size in a dataset, with the same smallest-set-of-smallest-rings code as the
            featurizer (Table 6: "rings in the data")
  runs      per run tag: number of runs, training time, epochs, parameters and the hyper-parameters stored in
            results.json (Methods 2.10 training time; Supplementary Table S3 for the retrained baselines)
  hardware  CPU and GPU of this machine (run it on the machine that trained the models)

Usage
  PYTHONPATH=<tebnet root> python check/paper_numbers.py rings --csv ../../models-baseline/deeptherm_data.csv
  python check/paper_numbers.py runs --runs benchmark_results_v153 benchmark_results_baselines
  python check/paper_numbers.py hardware
"""
from __future__ import annotations

import argparse
import json
import platform
from collections import defaultdict
from pathlib import Path

import numpy as np

# arguments that identify a run rather than configure the model
SKIP_ARGS = {"seed", "split_seed", "split_type", "member", "gpu", "out_dir", "csv", "physics_cache", "overwrite",
             "num_workers", "tag_suffix", "test_group", "test_smarts", "split_fractions", "outer_folds"}


def cmd_rings(args):
    import pandas as pd
    from rdkit import Chem, RDLogger
    from tebnet.data.featurizer import sssr_bond_rings      # same ring perception as the prior
    RDLogger.DisableLog("rdApp.*")
    df = pd.read_csv(args.csv)
    col = args.smiles_col or next(c for c in df.columns if str(c).strip().lower() in ("smiles", "smile"))
    labels = ["3", "4", "5", "6", "≥7"]
    rings = dict.fromkeys(labels, 0)
    mols = dict.fromkeys(labels, 0)
    n_ok = 0
    for s in df[col].astype(str):
        m = Chem.MolFromSmiles(s.strip())
        if m is None:
            continue
        n_ok += 1
        seen = set()
        for r in sssr_bond_rings(m):
            k = str(len(r)) if len(r) < 7 else "≥7"
            rings[k] += 1
            seen.add(k)
        for k in seen:
            mols[k] += 1
    print(f"{args.csv}: {n_ok} molecules parsed\n")
    print("| Ring size | Rings | Molecules with ≥ 1 | ≈ rings in an 81 % train split |")
    print("|---|---|---|---|")
    for k in labels:
        print(f"| {k} | {rings[k]} | {mols[k]} | {round(0.81 * rings[k])} |")


def cmd_runs(args):
    by_tag = defaultdict(list)
    for root in args.runs:
        for rp in sorted(Path(root).rglob("results.json")):
            try:
                res = json.loads(rp.read_text())
            except Exception:
                continue
            by_tag[res.get("tag", rp.parent.parent.parent.name)].append(res)
    if not by_tag:
        raise SystemExit(f"no results.json under {args.runs}")
    for tag in sorted(by_tag):
        rs = by_tag[tag]

        def stat(key):
            v = np.array([r[key] for r in rs if isinstance(r.get(key), (int, float))], dtype=float)
            return f"{v.mean():.0f} ± {v.std(ddof=1):.0f}" if len(v) > 1 else (f"{v[0]:.0f}" if len(v) else "n/a")

        t = np.array([r["training_time_sec"] for r in rs if isinstance(r.get("training_time_sec"), (int, float))])
        print(f"\n### {tag}  ({len(rs)} runs)")
        print(f"- training time per run: {stat('training_time_sec')} s"
              + (f" (total {t.sum() / 3600:.1f} h)" if len(t) else ""))
        print(f"- epochs trained: {stat('epochs_trained')}; best epoch: {stat('best_epoch')}; "
              f"parameters: {stat('n_params')}")
        a = rs[0].get("args", {})
        varying = sorted(k for k in a if k not in SKIP_ARGS and any(r.get("args", {}).get(k) != a[k] for r in rs))
        fixed = {k: v for k, v in sorted(a.items()) if k not in SKIP_ARGS and k not in varying}
        if fixed:
            print("- settings: " + ", ".join(f"{k}={v}" for k, v in fixed.items()))
        if varying:
            print("- differs between runs: " + ", ".join(varying))


def cmd_hardware(_args):
    cpu = platform.processor() or "?"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    print(f"CPU: {cpu} | Python {platform.python_version()} | {platform.platform()}")
    try:
        import torch
        print(f"torch {torch.__version__}, CUDA {torch.version.cuda}")
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {p.name}, {p.total_memory / 2**30:.0f} GB")
    except ImportError:
        print("torch not importable here")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rings")
    r.add_argument("--csv", required=True)
    r.add_argument("--smiles_col", default=None)
    u = sub.add_parser("runs")
    u.add_argument("--runs", nargs="+", required=True)
    sub.add_parser("hardware")
    args = ap.parse_args()
    {"rings": cmd_rings, "runs": cmd_runs, "hardware": cmd_hardware}[args.cmd](args)


if __name__ == "__main__":
    main()
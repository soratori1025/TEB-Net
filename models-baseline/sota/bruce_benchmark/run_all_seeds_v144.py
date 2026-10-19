"""
Orchestrate multi-seed TEB-Net v14.2 training on WUDILY-CHO dataset.

Runs 5 seeds, distributing across available GPUs.
Each seed uses the same data split (split_seed = init_seed, matching the
paper's single-split-per-seed protocol).
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import concurrent.futures
from pathlib import Path

PYTHON = "/home/nvsinh1/brats_segmentation/teb-net-model/.venv/bin/python3"
TRAIN_SCRIPT = str(Path(__file__).resolve().parent / "train_tebnet_v144.py")
SEEDS = [42, 123, 2024, 8888, 9999]
DEFAULT_GPUS = [0, 1]


def run_seed(seed: int, gpu: int, args):
    """Train a single seed."""
    cmd = [
        PYTHON, TRAIN_SCRIPT,
        "--seed", str(seed),
        "--split_seed", str(seed),
        "--csv", args.csv,
        "--split_dir", args.split_dir,
        "--out_dir", args.out_dir,
        "--max_epochs", str(args.max_epochs),
        "--patience", str(args.patience),
        "--batch_size", str(args.batch_size),
        "--gpu", str(gpu),
    ]
    print(f"[GPU {gpu}] Starting seed {seed}...")
    process = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if process.returncode != 0:
        print(f"[ERROR] Seed {seed} failed:\n{process.stdout[-2000:]}")
        return seed, False, process.stdout
    else:
        # Print the summary block
        lines = process.stdout.strip().split("\n")
        summary_start = None
        for i, line in enumerate(lines):
            if "TEBNet v14.2" in line and "Bruce Protocol" in line:
                summary_start = i
                break
        if summary_start is not None:
            print("\n".join(lines[summary_start:]))
        else:
            print(f"[OK] Seed {seed} finished successfully")
        return seed, True, process.stdout


def main():
    parser = argparse.ArgumentParser(description="Run all seeds for Bruce benchmark")
    bruce_root = Path(__file__).resolve().parent
    parser.add_argument("--csv", type=str,
                        default=str(bruce_root / "data" / "wudily_cho_processed.csv"))
    parser.add_argument("--split_dir", type=str,
                        default=str(bruce_root / "data" / "splits"))
    parser.add_argument("--out_dir", type=str,
                        default=str(bruce_root / "results_v144"))
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpus", type=str, default="0,1",
                        help="Comma-separated GPU IDs")
    parser.add_argument("--max_workers", type=int, default=None,
                        help="Max parallel workers (default: number of GPUs)")
    args = parser.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
    max_workers = args.max_workers or len(gpus)

    print(f"{'='*60}")
    print(f" Bruce Benchmark — TEB-Net v14.2 Multi-Seed Training")
    print(f"{'='*60}")
    print(f"  Seeds: {SEEDS}")
    print(f"  GPUs: {gpus}")
    print(f"  Workers: {max_workers}")
    print(f"  Max epochs: {args.max_epochs}")
    print(f"  Patience: {args.patience}")
    print(f"{'='*60}\n")

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, seed in enumerate(SEEDS):
            gpu = gpus[i % len(gpus)]
            future = executor.submit(run_seed, seed, gpu, args)
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            seed, success, output = future.result()
            results[seed] = success

    # Summary
    print(f"\n{'='*60}")
    print(f" Training Summary")
    print(f"{'='*60}")
    n_ok = sum(1 for v in results.values() if v)
    n_fail = sum(1 for v in results.values() if not v)
    for seed in SEEDS:
        status = "✓ OK" if results.get(seed) else "✗ FAILED"
        print(f"  Seed {seed}: {status}")
    print(f"\n  {n_ok}/{len(SEEDS)} succeeded, {n_fail} failed")
    if n_ok == len(SEEDS):
        print(f"\n  All seeds done! Run evaluate.py to generate the report.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

"""Precompute TEB-Net v15 physics features for thermochemistry datasets.

Replaces the old symmetry/linearity script. Differences:
  * Does NOT rewrite the raw dataset CSV any more. Results go to a cache file
    next to it:  <dataset_stem>.physics_v15.csv  (columns: smiles, features_version,
    + one named column per feature in featurizer.V15_NAMES).
    ThermochemDataset.from_csv(..., physics_cache="auto") picks it up automatically.
  * Single source of truth: features come from tebnet.data.featurizer.compute_v15_vector,
    i.e. exactly what the model uses when no cache exists (the old script wrote
    [log σ, N, is_linear, M] while the featurizer expected [log σ, is_linear, N, radical_e],
    and the v14.4 pipeline never read that column anyway).
  * Unique SMILES only, multiprocessing, resumable (re-run continues where it stopped).

Usage
  python compute_physics_features.py --csv /path/deeptherm_data_clean.csv /path/wudily.csv
  python compute_physics_features.py --dataset all            # burcat / rmg / qm9 presets next to this file
  python compute_physics_features.py --csv data.csv --no-3d   # skip ETKDG moments of inertia (fast, strict 2-D)
If `tebnet` is not importable from here, pass --tebnet-root /path/that/contains/tebnet
"""
from __future__ import annotations
import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def _setup_paths() -> None:
    """Make `tebnet` (and the data/ helper modules it imports) importable, also in spawned workers."""
    cands: list[Path] = []
    env = os.environ.get("TEBNET_ROOT")
    if env:
        cands.append(Path(env))
    here = Path(__file__).resolve()
    for p in [here.parent, *list(here.parents)[:4]]:
        cands.append(p)
        try:
            cands += [c for c in p.iterdir() if c.is_dir() and (c / "tebnet").is_dir()]
        except OSError:
            pass
    for c in cands:
        if (c / "tebnet").is_dir() and str(c) not in sys.path:
            sys.path.insert(0, str(c))
            break
    for extra in (here.parent, here.parent.parent, here.parent.parent / "data"):
        if str(extra) not in sys.path:
            sys.path.append(str(extra))


_setup_paths()


def _worker(item: tuple[str, bool]):
    from tebnet.data.featurizer import compute_v15_from_smiles
    smi, use_3d = item
    return smi, compute_v15_from_smiles(smi, use_3d=use_3d)


def process_csv(csv_path: Path, smiles_col: str, n_workers: int, use_3d: bool,
                save_interval: int, overwrite: bool) -> None:
    from tebnet.data.featurizer import V15_NAMES, FEATURES_VERSION, physics_cache_path

    print(f"\n==========================================\n[v15 PHYSICS] {csv_path.name}\n"
          f"==========================================")
    if not csv_path.exists():
        print(f"[WARNING] File not found: {csv_path}")
        return
    df = pd.read_csv(csv_path)
    if smiles_col not in df.columns:
        print(f"[ERROR] Column '{smiles_col}' not found in {csv_path}")
        return
    smiles = (df[smiles_col].dropna().astype(str).str.strip())
    smiles = sorted(set(s for s in smiles if s))
    out_path = physics_cache_path(csv_path)
    print(f"Records: {len(df):,} | unique SMILES: {len(smiles):,} | 3-D: {use_3d} | cache: {out_path.name}")

    rows: dict[str, list[float]] = {}
    if out_path.exists() and not overwrite:
        old = pd.read_csv(out_path)
        ok_version = ("features_version" in old.columns
                      and (old["features_version"].astype(str) == FEATURES_VERSION).all()
                      and all(n in old.columns for n in V15_NAMES))
        if ok_version:
            for s, vec in zip(old["smiles"].astype(str), old[V15_NAMES].to_numpy(np.float64)):
                rows[s] = vec.tolist()
            print(f"Resuming: {len(rows):,} species already in cache")
        else:
            print("Existing cache has another feature version — recomputing everything")

    todo = [s for s in smiles if s not in rows]
    failed: list[str] = []

    def _save():
        keys = sorted(rows)
        out = pd.DataFrame(np.array([rows[k] for k in keys], dtype=np.float64).reshape(len(keys), len(V15_NAMES)),
                           columns=V15_NAMES)
        out.insert(0, "features_version", FEATURES_VERSION)
        out.insert(0, "smiles", keys)
        tmp = out_path.with_suffix(".tmp.csv")
        out.to_csv(tmp, index=False)
        os.replace(tmp, out_path)

    t0 = time.time()
    if todo:
        chunksize = max(1, min(64, len(todo) // (n_workers * 8) or 1))
        n_since_save = 0
        with mp.Pool(processes=n_workers) as pool:
            for smi, vec in tqdm(pool.imap_unordered(_worker, [(s, use_3d) for s in todo], chunksize=chunksize),
                                 total=len(todo), desc=csv_path.name):
                if vec is None:
                    failed.append(smi)
                else:
                    rows[smi] = vec
                n_since_save += 1
                if n_since_save >= save_interval:
                    _save()
                    n_since_save = 0
    _save()

    arr = np.array([rows[k] for k in rows]) if rows else np.zeros((0, len(V15_NAMES)))
    col = {n: i for i, n in enumerate(V15_NAMES)}
    if len(arr):
        print(f"Done in {time.time() - t0:.1f}s | ok {len(rows):,} | invalid SMILES {len(failed)}")
        print(f"  radicals     : {(arr[:, col['n_rad_e']] > 0).sum():,}")
        print(f"  cyclic       : {(arr[:, col['n_rings']] > 0).sum():,}  "
              f"(polycyclic {(arr[:, col['n_rings']] >= 2).sum():,})")
        print(f"  linear       : {(arr[:, col['is_linear']] > 0.5).sum():,} | "
              f"monatomic {(arr[:, col['is_monatomic']] > 0.5).sum():,}")
        print(f"  σ > 1        : {(arr[:, col['log_sigma']] > 0).sum():,}")
        if use_3d:
            print(f"  3-D embedding failed: {(arr[:, col['moi_valid']] < 0.5).sum():,}")
    if failed:
        print(f"  first invalid SMILES: {failed[:5]}")
    print(f"[SUCCESS] {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Precompute TEB-Net v15 physics features (cache CSV per dataset).")
    ap.add_argument("--csv", nargs="+", default=None, help="dataset CSV file(s)")
    ap.add_argument("--dataset", choices=["burcat", "rmg", "qm9", "all"], default=None,
                    help="presets next to this script (old behaviour)")
    ap.add_argument("--smiles-col", default="smiles")
    ap.add_argument("--n-workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--save-interval", type=int, default=2000)
    ap.add_argument("--no-3d", action="store_true", help="skip ETKDG moments of inertia")
    ap.add_argument("--overwrite", action="store_true", help="ignore an existing cache")
    ap.add_argument("--tebnet-root", default=None, help="directory that contains the tebnet/ package")
    args = ap.parse_args()

    if args.tebnet_root:
        os.environ["TEBNET_ROOT"] = str(Path(args.tebnet_root).resolve())
        sys.path.insert(0, os.environ["TEBNET_ROOT"])
    try:
        import tebnet.data.featurizer  # noqa: F401
    except ImportError as e:
        sys.exit(f"Cannot import tebnet ({e}). Pass --tebnet-root /path/containing/tebnet")

    data_dir = Path(__file__).resolve().parent
    presets = {
        "burcat": data_dir / "burcat_with_smiles_valid.csv",
        "rmg": data_dir / "rmg_thermo.csv",
        "qm9": data_dir / "qm9_thermo.csv",
    }
    targets: list[Path] = []
    if args.csv:
        targets += [Path(p).resolve() for p in args.csv]
    if args.dataset:
        targets += list(presets.values()) if args.dataset == "all" else [presets[args.dataset]]
    if not targets:
        ap.error("give --csv and/or --dataset")

    print(f"CPU workers: {args.n_workers} | 3-D: {not args.no_3d}")
    for p in targets:
        process_csv(p, args.smiles_col, args.n_workers, not args.no_3d, args.save_interval, args.overwrite)


if __name__ == "__main__":
    main()
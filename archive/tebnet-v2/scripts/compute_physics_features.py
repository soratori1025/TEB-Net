import pandas as pd
import argparse
import sys
import multiprocessing
from pathlib import Path
from tqdm import tqdm
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from tebnet.data.featurizer import compute_v15_from_smiles, V15_NAMES, physics_cache_path
def process_smiles(args):
    idx, smiles = args
    return idx, compute_v15_from_smiles(smiles, use_3d=True)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smiles_col", default="smiles")
    args = parser.parse_args()
    out_path = physics_cache_path(args.csv)
    if out_path.exists() and not args.overwrite:
        print(f"File {out_path} already exists. Skipping.")
        return
    df = pd.read_csv(args.csv)
    smiles_list = df[args.smiles_col].tolist()
    print(f"Computing physics features for {len(smiles_list)} molecules...")
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    results = [None] * len(smiles_list)
    tasks = [(i, s) for i, s in enumerate(smiles_list)]
    for idx, res in tqdm(pool.imap_unordered(process_smiles, tasks), total=len(tasks)):
        results[idx] = res
    pool.close()
    for i in range(len(results)):
        if results[i] is None:
            results[i] = [0.0] * len(V15_NAMES)
    res_df = pd.DataFrame(results, columns=V15_NAMES)
    res_df["smiles"] = smiles_list
    res_df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
if __name__ == "__main__":
    main()

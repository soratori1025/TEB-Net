"""D-MPNN (chemprop) predictions on the five homologous series and the DeepTherm test set.

Uses the chemprop v2 checkpoint of ``models-baseline/sota/chemprop_repo/chemprop_dmpnn_random``
replicate 0, which was trained on exactly the DeepTherm random split 0 of the other baselines
(splits/splits_random.json[0] == the TEB-Net split; verified by the 501/501 test-set overlap).
Writes ``<out_dir>/preds_dmpnn_chemprop.csv`` in the common format
(``smiles, H_kcal, S_cal, Cp300_cal``) and ``<out_dir>/verify_dmpnn_chemprop.json`` with the
in-distribution test MAE recovered from the checkpoint.

    python models-baseline/traditional_models/ch2_extrapolation_dmpnn.py [--device cpu]
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
BENCH_ROOT = HERE.parent
REPO_ROOT = BENCH_ROOT.parent
CKPT = REPO_ROOT / "models-baseline" / "sota" / "chemprop_repo" / "chemprop_dmpnn_random" / "replicate_0" / "model_0" / "best.pt"
DT_TARGETS = ["Hf_298.15K_kcal/mol", "S_298.15K_cal/(molK)", "cp_300K_cal/(molK)", "cp_400K_cal/(molK)",
              "cp_500K_cal/(molK)", "cp_600K_cal/(molK)", "cp_800K_cal/(molK)", "cp_1000K_cal/(molK)",
              "cp_1500K_cal/(molK)"]
def chemprop_predict(csv_in: Path, csv_out: Path, ckpt: Path, accelerator: str = "cpu", devices: str = "1") -> None:
    cmd = ["chemprop", "predict", "--test-path", str(csv_in), "--smiles-columns", "smiles",
           "--model-paths", str(ckpt), "--preds-path", str(csv_out),
           "--accelerator", accelerator, "--devices", devices]
    subprocess.run(cmd, check=True)
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series_csv", default=str(REPO_ROOT / "models" / "tebnet-v3" / "check" / "homologous_series.csv"))
    ap.add_argument("--out_dir", default=str(BENCH_ROOT / "ch2_extrapolation"))
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ser = pd.read_csv(args.series_csv)
        smi = ser["smiles"].astype(str).str.strip().tolist()
        inp = tmp / "homologs.csv"
        pd.DataFrame({"smiles": smi}).to_csv(inp, index=False)
        raw = tmp / "homologs_preds.csv"
        chemprop_predict(inp, raw, CKPT, "cpu" if args.device == "cpu" else "gpu")
        pr = pd.read_csv(raw)
        out = pd.DataFrame({"smiles": pr["smiles"], "H_kcal": pr[DT_TARGETS[0]],
                            "S_cal": pr[DT_TARGETS[1]], "Cp300_cal": pr[DT_TARGETS[2]]})
        out.to_csv(out_dir / "preds_dmpnn_chemprop.csv", index=False, float_format="%.4f")
        print(f"wrote {out_dir / 'preds_dmpnn_chemprop.csv'} ({len(out)} species)")
        data = pd.read_csv(BENCH_ROOT / "deeptherm_data.csv")
        d_in = tmp / "deeptherm.csv"
        data.to_csv(d_in, index=False)
        raw_all = tmp / "deeptherm_preds.csv"
        chemprop_predict(d_in, raw_all, CKPT, "cpu" if args.device == "cpu" else "gpu")
        pa = pd.read_csv(raw_all)
        gin = json.loads((BENCH_ROOT / "benchmark_results_baselines" / "gnn-gin_all_L4_d300_sum" /
                          "random0" / "member0_seed0" / "results.json").read_text())
        te = np.asarray(gin["test_indices"])
        rep = {"tag": "dmpnn_chemprop", "checkpoint": str(CKPT.relative_to(REPO_ROOT)), "split": "random0"}
        for k, c in (("H", DT_TARGETS[0]), ("S", DT_TARGETS[1]), ("Cp300", DT_TARGETS[2])):
            true = data[c].to_numpy(float)[te]
            pred = pa[c].to_numpy(float)[te]
            rep[f"test_mae_{k}"] = float(np.nanmean(np.abs(pred - true)))
        (out_dir / "verify_dmpnn_chemprop.json").write_text(json.dumps(rep, indent=2))
        print("test MAE:", {k: round(v, 3) for k, v in rep.items() if k.startswith("test_mae")})
if __name__ == "__main__":
    main()

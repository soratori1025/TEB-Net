"""Re-run the NASA-7 export and the high-temperature validation in one go.

Kept as a small Python entry point next to run_pipeline.sh for people who would rather not
use a shell script. Both call exactly the commands documented in REPRODUCE.md, including
`--agg trimmed`: the default of nasa7_check.py is the weighted mean, which does not match
the manuscript.
"""
from __future__ import annotations
import glob
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
PY = sys.executable
H_GLOB = str(ROOT / "benchmark_results_deeptherm/thermo_physics_chebyshev_3d_grouped_d5_global_w1-0-0_mon-h_sp-v2_exact_splitH/random0/member*_seed*")
SCP_GLOB = str(ROOT / "benchmark_results_deeptherm/thermo_physics_chebyshev_3d_grouped_d5_global_w0-1.5-1_mon-scp_sp-v2_exact_splitSCp/random0/member*_seed*")
DAT = ROOT / "logs_v3/c5_rmg.dat"
h_runs = sorted(glob.glob(H_GLOB))
scp_runs = sorted(glob.glob(SCP_GLOB))
if not h_runs or not scp_runs:
    raise SystemExit("trained run directories not found; see README (checkpoints are not in git)")
cmd1 = [PY, "check/nasa7_check.py", "--h_runs", *h_runs, "--scp_runs", *scp_runs,
        "--smiles_csv", "rmg_cp_reference.csv", "--complete_high", "--agg", "trimmed",
        "--write_dat", str(DAT)]
print("running:", " ".join(cmd1[:3]), f"({len(h_runs)} + {len(scp_runs)} members)")
subprocess.run(cmd1, cwd=ROOT, check=True)
cmd2 = [PY, "check/high_t_validation.py", str(DAT), "--reference", "rmg_cp_reference.csv",
        "--as_trained", str(DAT.with_suffix(".as_trained.dat")), "--out_csv", "high_t_validation.csv"]
print("running:", " ".join(cmd2))
with open(ROOT / "high_t_validation.txt", "w") as fh:
    subprocess.run(cmd2, cwd=ROOT, stdout=fh, check=True)
cmd3 = [PY, "check/plot_cp_completion.py", str(DAT), "--reference", "rmg_cp_reference.csv",
        "--error_panel", "--out", "fig_cp_completion"]
print("running:", " ".join(cmd3))
subprocess.run(cmd3, cwd=ROOT, check=True)
print("done: logs_v3/c5_rmg.dat, high_t_validation.csv/.txt, fig_cp_completion.pdf/.png")

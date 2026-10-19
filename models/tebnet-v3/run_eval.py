import os, subprocess, glob

runs = glob.glob("benchmark_results_deeptherm/thermo_physics_chebyshev_3d_grouped_d5_global_sp-v2_exact/random0/*")
cmd1 = ["python", "check/nasa7_check.py", "--h_runs"] + runs + ["--smiles_csv", "rmg_cp_reference.csv", "--complete_high", "--write_dat", "c5_rmg.dat"]
print(f"Running cmd1: {' '.join(cmd1)}")
subprocess.run(cmd1, check=True)

cmd2 = ["python", "check/high_t_validation.py", "c5_rmg.dat", "--reference", "rmg_cp_reference.csv", "--deeptherm_csv", "../../models-baseline/deeptherm_data.csv"]
print(f"Running cmd2: {' '.join(cmd2)}")
with open("high_t_validation.txt", "w") as f:
    subprocess.run(cmd2, stdout=f, check=True)

cmd3 = ["python", "check/plot_cp_completion.py", "c5_rmg.dat", "--reference", "rmg_cp_reference.csv", "--error_panel", "--out", "fig_cp_completion"]
print(f"Running cmd3: {' '.join(cmd3)}")
subprocess.run(cmd3, check=True)

print("All done!")

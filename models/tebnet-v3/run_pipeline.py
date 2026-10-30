import os
import subprocess
import time
from pathlib import Path
def run_cmd(cmd, env, log_file):
    print(f"Queueing: {cmd} > {log_file}")
    with open(log_file, "w") as f:
        proc = subprocess.Popen(cmd, shell=True, executable='/bin/bash', env=env, stdout=f, stderr=subprocess.STDOUT)
    return proc
def train_split(base_cmd, gpu, log_prefix):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    h_cmd = f"{base_cmd} --loss_weights 1 0 0 --monitor h --tag_suffix _splitH"
    scp_cmd = f"{base_cmd} --loss_weights 0 1.5 1 --monitor scp --tag_suffix _splitSCp"
    def run_if_needed(cmd, log_file):
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                content = f.read()
            if "Results:" in content or "Total time:" in content or "per-T Cp MAE" in content:
                print(f"Skipping (already done): {log_file}")
                return None
        return run_cmd(cmd, env, log_file)
    p1 = run_if_needed(h_cmd, f"{log_prefix}_H.log")
    p2 = run_if_needed(scp_cmd, f"{log_prefix}_SCp.log")
    return p1, p2
def main():
    os.makedirs("logs_v3/R_runs", exist_ok=True)
    CSV = "/home/nvsinh1/brats_segmentation/teb-net-model/models-baseline/deeptherm_data.csv"
    if not os.path.exists(CSV):
        CSV = "/home/nvsinh1/brats_segmentation/teb-net-model/models-s3/tebnet-v3/deeptherm_data.csv"
    tasks = []
    for m in range(10):
        tasks.append((f"python scripts/train.py --seed {m} --member {m} --split_seed 0 --csv {CSV} --s_prior v2_exact --overwrite", f"logs_v3/R_runs/R12_m{m}_s0"))
    for s in range(1, 5):
        tasks.append((f"python scripts/train.py --seed 0 --member 0 --split_seed {s} --csv {CSV} --s_prior v2_exact --overwrite", f"logs_v3/R_runs/R1_m0_s{s}"))
    for s in range(2):
        tasks.append((f"python scripts/train.py --seed {s} --member {s} --split_type complexity --csv {CSV} --s_prior v2_exact --overwrite", f"logs_v3/R_runs/R3_comp_m{s}"))
    for g in ["Ether group", "Epoxide group", "Enol group"]:
        g_log = g.replace(" ", "_")
        tasks.append((f"python scripts/train.py --seed 0 --split_type functional_group --test_group '{g}' --csv {CSV} --s_prior v2_exact --overwrite", f"logs_v3/R_runs/R4_{g_log}"))
    for frac in [0.1, 0.2, 0.4, 0.6, 0.8]:
        tasks.append((f"python scripts/train.py --seed 0 --split_seed 0 --split_fractions {frac} 0.1 {1.0 - frac - 0.1:.1f} --csv {CSV} --s_prior v2_exact --tag_suffix _frac{frac} --overwrite", f"logs_v3/R_runs/R5_frac{frac}"))
    tasks.append((f"python scripts/train.py --seed 0 --split_seed 0 --baseline none --csv {CSV} --overwrite", f"logs_v3/R_runs/R6_DT_rand0_noprior"))
    tasks.append((f"python scripts/train.py --seed 0 --split_type complexity --baseline none --csv {CSV} --overwrite", f"logs_v3/R_runs/R6_DT_comp_noprior"))
    tasks.append((f"python scripts/train.py --seed 0 --split_seed 0 --split_fractions 0.1 0.1 0.8 --baseline none --csv {CSV} --overwrite", f"logs_v3/R_runs/R6_DT_frac0.1_noprior"))
    for ureel_set in ["carbenium_hc", "radical_hc", "stable_hc"]:
        ureel_csv = f"/home/nvsinh1/brats_segmentation/teb-net-model/data/splits/{ureel_set}.csv"
        tasks.append((f"python scripts/train_cv.py --csv {ureel_csv} --prior_only --s_prior v2_exact --h_unit j", f"logs_v3/R_runs/R7_{ureel_set}_prior"))
        for k in range(10):
            tasks.append((f"python scripts/train_cv.py --csv {ureel_csv} --outer_folds {k} --s_prior v2_exact --h_unit j", f"logs_v3/R_runs/R7_{ureel_set}_k{k}"))
    tasks.append((f"python scripts/train_cv.py --csv /home/nvsinh1/brats_segmentation/teb-net-model/data/splits/stable_hc.csv --outer_folds 0 --baseline none --h_unit j", f"logs_v3/R_runs/R6_ureel_stable_noprior"))
    tasks.append((f"python scripts/train_cv.py --csv /home/nvsinh1/brats_segmentation/teb-net-model/data/splits/radical_hc.csv --outer_folds 0 --baseline none --h_unit j", f"logs_v3/R_runs/R6_ureel_radical_noprior"))
    for i in range(0, len(tasks), 2):
        batch = tasks[i:i+2]
        ps = []
        for j, (cmd, log) in enumerate(batch):
            gpu = j % 2
            if "prior_only" in cmd:
                env = os.environ.copy()
                log_file = log + ".log"
                if os.path.exists(log_file) and "time:" in open(log_file).read():
                    print(f"Skipping (already done): {log_file}")
                else:
                    p = run_cmd(cmd, env, log_file)
                    ps.append(p)
            else:
                p1, p2 = train_split(cmd, gpu, log)
                if p1: ps.append(p1)
                if p2: ps.append(p2)
        for p in ps:
            if p: p.wait()
if __name__ == '__main__':
    main()

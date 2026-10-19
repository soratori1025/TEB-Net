import os
import subprocess
import time

def run_cmd(cmd, env=None, log_file=None):
    print(f"Running: {cmd}")
    if log_file:
        with open(log_file, "w") as f:
            proc = subprocess.Popen(cmd, shell=True, executable='/bin/bash', env=env, stdout=f, stderr=f)
    else:
        proc = subprocess.Popen(cmd, shell=True, executable='/bin/bash', env=env)
    return proc

def train_split(args_str, gpu, log_prefix):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    h_cmd = f"python scripts/train.py {args_str} --loss_weights 1 0 0 --monitor h --tag_suffix _splitH"
    scp_cmd = f"python scripts/train.py {args_str} --loss_weights 0 1.5 1 --monitor scp --tag_suffix _splitSCp"
    
    p1 = run_cmd(h_cmd, env, f"{log_prefix}_H.log")
    p2 = run_cmd(scp_cmd, env, f"{log_prefix}_SCp.log")
    return p1, p2

def main():
    os.makedirs("logs_v3/orchestration", exist_ok=True)
    CSV = "/home/nvsinh1/brats_segmentation/teb-net-model/models-s3/tebnet-v3/deeptherm_data.csv"
    
    procs = []
    
    # R1 & R2: Random splits
    # R2 needs seeds 0-9 on split 0. R1 needs member 0 on splits 0-4.
    # So we need (seed 0..9, split 0) + (seed 0, split 1..4)
    tasks = []
    for m in range(10):
        tasks.append(f"--seed {m} --member {m} --split_seed 0 --csv {CSV} --s_prior v2_exact --overwrite")
    for s in range(1, 5):
        tasks.append(f"--seed 0 --member 0 --split_seed {s} --csv {CSV} --s_prior v2_exact --overwrite")
        
    # Run tasks 2 per GPU (4 concurrently)
    for i in range(0, len(tasks), 2):
        batch = tasks[i:i+2]
        ps = []
        for j, t in enumerate(batch):
            gpu = j % 2
            log = f"logs_v3/orchestration/R12_task_{i+j}"
            p1, p2 = train_split(t, gpu, log)
            ps.extend([p1, p2])
        for p in ps:
            p.wait()

if __name__ == '__main__':
    main()

import os
import subprocess
import shutil
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    print("Starting Chemprop HPO with hpopt...")
    train_csv = PROJECT_ROOT / "data/chemprop_data/stable_random_train.csv"
    val_csv = PROJECT_ROOT / "data/chemprop_data/stable_random_test.csv"
    
    save_dir = PROJECT_ROOT / "sota/chemprop_repo/checkpoints_chemprop/optuna_study"
    os.makedirs(save_dir, exist_ok=True)
    
    hpo_cmd = [
        str(PROJECT_ROOT / ".venv/bin/chemprop"), "hpopt",
        "--data-path", str(train_csv),
        "--epochs", "30",
        "--raytune-num-samples", "15",
        "--search-parameter-keywords", "basic",
        "--hpopt-save-dir", str(save_dir)
    ]
    
    print(f"Running command: {' '.join(hpo_cmd)}")
    subprocess.run(hpo_cmd)
    
    # Check for best config file
    best_config = None
    if os.path.exists(save_dir):
        for root, dirs, files in os.walk(save_dir):
            for file in files:
                if file.endswith('.toml') or file.endswith('.json'):
                    best_config = Path(root) / file
                    break
    
    if best_config:
        dest_dir = PROJECT_ROOT / "configs" / "optuna"
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy(best_config, dest_dir / f"best_chemprop_params{best_config.suffix}")
        print(f"Saved best params to {dest_dir}/best_chemprop_params{best_config.suffix}")
    else:
        print("Could not find the best config file from Chemprop hpopt output.")

if __name__ == "__main__":
    main()

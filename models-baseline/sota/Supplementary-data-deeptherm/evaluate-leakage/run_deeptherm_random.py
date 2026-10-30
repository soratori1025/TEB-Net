import sys
import traceback
import json
from pathlib import Path
import os
sys.path.insert(0, "/home/nvsinh1/brats_segmentation/teb-net-model/models_evaluate/sota/DeepTherm")
from deeptherm.data import read_csv_dataset
from deeptherm.training import TrainingConfig, train_ensemble
try:
    print("Reading data...")
    frame, task_names = read_csv_dataset(
        "/home/nvsinh1/brats_segmentation/teb-net-model/paper-consistancy/benchmark-evaluate/deeptherm_data_clean.csv",
        smiles_column="smiles",
        target_columns=None
    )
    print("Task names:", task_names)
    config = TrainingConfig(
        epochs=100,
        batch_size=128,
        device="cuda"
    )
    save_path = Path("/home/nvsinh1/brats_segmentation/teb-net-model/models_evaluate/sota/DeepTherm/runs/test_random")
    print(f"Starting train_ensemble (Ensemble Size: 10) saving to {save_path}...")
    results = train_ensemble(
        frame=frame,
        smiles_column="smiles",
        task_names=task_names,
        save_dir=save_path,
        descriptor_kind="morgan",
        model_options={
            "hidden_size": 300,
            "depth": 3,
            "num_attention_heads": 4,
            "ffn_hidden_size": 300,
            "ffn_num_layers": 2,
            "dropout": 0.1,
        },
        training_config=config,
        split_strategy="random",
        split_fractions=(0.81, 0.09, 0.10),
        seed=42,
        ensemble_size=10
    )
    print("Training finished successfully.")
    import numpy as np
    from deeptherm.training import predict_ensemble
    test_smiles = [frame.iloc[idx]['smiles'] for idx in range(len(frame))]
    metrics_path = save_path / "ensemble_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        test_maes = [m['test']['mae'] for m in metrics]
        print(f"Individual Model Test MAEs: {test_maes}")
        print(f"Average of Individual Test MAEs: {np.mean(test_maes):.4f}")
        print("Note: True ensemble MAE requires ensembling predictions first, which might be lower.")
except Exception as e:
    print("Error occurred!")
    traceback.print_exc()

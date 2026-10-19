import sys
import traceback
from pathlib import Path

# Add DeepTherm to python path
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
        epochs=1,
        batch_size=128,
        device="cuda"
    )
    
    print("Starting train_ensemble...")
    results = train_ensemble(
        frame=frame,
        smiles_column="smiles",
        task_names=task_names,
        save_dir=Path("/home/nvsinh1/brats_segmentation/teb-net-model/models_evaluate/sota/DeepTherm/runs/test_random"),
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
        ensemble_size=1
    )
    print("Training finished:", results)
except Exception as e:
    print("Error occurred!")
    traceback.print_exc()

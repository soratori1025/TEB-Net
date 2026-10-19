import sys
sys.path.insert(0, "models_evaluate/sota/DeepTherm")
from deeptherm.data import read_csv_dataset, split_indices
frame, task_names = read_csv_dataset("paper-consistancy/benchmark-evaluate/deeptherm_data_clean.csv", "smiles", None)
print("Task names:", task_names)
print("Frame shape:", frame.shape)
train, valid, test = split_indices(frame["smiles"].tolist(), "random", (0.8, 0.1, 0.1), 42)
print("Split sizes:", len(train), len(valid), len(test))

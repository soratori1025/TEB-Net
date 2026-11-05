import pandas as pd
import json
import os

df = pd.read_csv('data/splits/stable_hc.csv')
TARGETS = ['dH_f_298', 'S_298', 'Cp_300', 'Cp_400', 'Cp_500', 'Cp_600', 'Cp_800', 'Cp_1000']
COLUMNS = ['canonical_smiles'] + TARGETS

os.makedirs('data/chemprop_data', exist_ok=True)

# 1. Random Split
with open('data/splits/stable_5fold.json', 'r') as f:
    folds = json.load(f)
test_idx = folds[0]
train_idx = list(set(range(len(df))) - set(test_idx))

train_df = df.iloc[train_idx][COLUMNS].rename(columns={'canonical_smiles': 'smiles'})
test_df = df.iloc[test_idx][COLUMNS].rename(columns={'canonical_smiles': 'smiles'})
train_df.to_csv('data/chemprop_data/stable_random_train.csv', index=False)
test_df.to_csv('data/chemprop_data/stable_random_test.csv', index=False)

# 2. Scaffold Split
with open('data/splits/stable_scaffold_split.json', 'r') as f:
    scaffold = json.load(f)
train_df = df.iloc[scaffold['train']][COLUMNS].rename(columns={'canonical_smiles': 'smiles'})
test_df = df.iloc[scaffold['test']][COLUMNS].rename(columns={'canonical_smiles': 'smiles'})
train_df.to_csv('data/chemprop_data/stable_scaffold_train.csv', index=False)
test_df.to_csv('data/chemprop_data/stable_scaffold_test.csv', index=False)

# 3. Size Split
with open('data/splits/stable_size_split.json', 'r') as f:
    size_s = json.load(f)
train_df = df.iloc[size_s['train']][COLUMNS].rename(columns={'canonical_smiles': 'smiles'})
test_df = df.iloc[size_s['test']][COLUMNS].rename(columns={'canonical_smiles': 'smiles'})
train_df.to_csv('data/chemprop_data/stable_size_train.csv', index=False)
test_df.to_csv('data/chemprop_data/stable_size_test.csv', index=False)

print("Chemprop data prepped.")

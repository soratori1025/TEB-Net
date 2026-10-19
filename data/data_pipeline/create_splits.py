import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import os

df = pd.read_csv('data/hc_thermo_canonical.csv')

def classify_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 'unknown'
    num_radicals = sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())
    charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    
    if charge > 0:
        return 'carbenium'
    elif num_radicals > 0:
        return 'radical'
    else:
        return 'stable'

def get_heavy_atom_count(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return 0
    return mol.GetNumHeavyAtoms()

def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return ''
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return scaffold
    except:
        return ''

df['class'] = df['canonical_smiles'].apply(classify_molecule)
df['heavy_atoms'] = df['canonical_smiles'].apply(get_heavy_atom_count)
df['scaffold'] = df['canonical_smiles'].apply(get_scaffold)

stable_df = df[df['class'] == 'stable'].copy().reset_index(drop=True)
radical_df = df[df['class'] == 'radical'].copy().reset_index(drop=True)
carbenium_df = df[df['class'] == 'carbenium'].copy().reset_index(drop=True)

print(f"Total: {len(df)}")
print(f"Stable: {len(stable_df)}")
print(f"Radicals: {len(radical_df)}")
print(f"Carbenium: {len(carbenium_df)}")

# Create output dir
os.makedirs('data/splits', exist_ok=True)

stable_df.to_csv('data/splits/stable_hc.csv', index=False)
radical_df.to_csv('data/splits/radical_hc.csv', index=False)
carbenium_df.to_csv('data/splits/carbenium_hc.csv', index=False)

def create_random_kfold(df_name, df_data, k=5):
    np.random.seed(42)
    indices = np.random.permutation(len(df_data))
    fold_sizes = np.full(k, len(df_data) // k, dtype=int)
    fold_sizes[:len(df_data) % k] += 1
    
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append(indices[start:stop])
        current = stop
        
    # Save splits as JSON list of lists (test indices)
    import json
    with open(f'data/splits/{df_name}_5fold.json', 'w') as f:
        json.dump([fold.tolist() for fold in folds], f)

create_random_kfold('stable', stable_df)
create_random_kfold('radical', radical_df)
create_random_kfold('carbenium', carbenium_df)

# Scaffold Split for Stable HC (80/20)
# Group by scaffold, sort by size of group (descending), then assign to train/test
stable_df['scaffold'] = stable_df['scaffold'].fillna('')
scaffold_sets = stable_df.groupby('scaffold').indices
scaffolds = list(scaffold_sets.keys())
scaffolds.sort(key=lambda s: len(scaffold_sets[s]), reverse=True)

train_indices = []
test_indices = []

# For the empty scaffold (acyclics), split them randomly 80/20
if '' in scaffold_sets:
    empty_idx = list(scaffold_sets[''])
    np.random.seed(42)
    np.random.shuffle(empty_idx)
    split_point = int(0.8 * len(empty_idx))
    train_indices.extend(empty_idx[:split_point])
    test_indices.extend(empty_idx[split_point:])
    scaffolds.remove('')

for s in scaffolds:
    if len(train_indices) < 0.8 * len(stable_df):
        train_indices.extend(scaffold_sets[s])
    else:
        test_indices.extend(scaffold_sets[s])

import json
with open('data/splits/stable_scaffold_split.json', 'w') as f:
    json.dump({'train': [int(i) for i in train_indices], 'test': [int(i) for i in test_indices]}, f)

# Size Extrapolation Split for Stable HC
size_train = stable_df[stable_df['heavy_atoms'] <= 8].index.tolist()
size_test = stable_df[stable_df['heavy_atoms'] >= 9].index.tolist()

with open('data/splits/stable_size_split.json', 'w') as f:
    json.dump({'train': [int(i) for i in size_train], 'test': [int(i) for i in size_test]}, f)

print(f"Scaffold Split: {len(train_indices)} train, {len(test_indices)} test")
print(f"Size Split: {len(size_train)} train (<=8 heavy atoms), {len(size_test)} test (>=9 heavy atoms)")

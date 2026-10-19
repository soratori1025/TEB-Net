import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import Counter

def generate_murcko_scaffold(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "invalid"
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return scaffold if scaffold else "acyclic"
    except:
        return "invalid"

df = pd.read_csv("/home/nvsinh1/brats_segmentation/teb-net-model/paper-consistancy/benchmark-evaluate/deeptherm_data_clean.csv")
df['murcko'] = df['smiles'].apply(generate_murcko_scaffold)

# Filter out acyclic and invalid
cyclic_scaffolds = df[df['murcko'] != 'acyclic']['murcko'].tolist()
total_cyclic = len(cyclic_scaffolds)

counter = Counter(cyclic_scaffolds)
print(f"Total cyclic molecules: {total_cyclic}")
print(f"Number of unique cyclic scaffolds: {len(counter)}")
print("\nTop 10 most common scaffolds:")
for scaffold, count in counter.most_common(10):
    print(f"- {scaffold}: {count} ({count/total_cyclic*100:.2f}%)")

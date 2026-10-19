import pandas as pd
from rdkit import Chem
from rdkit.Chem import inchi
import os

def process_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None
        canonical = Chem.MolToSmiles(mol)
        inchikey = inchi.MolToInchiKey(mol)
        return canonical, inchikey
    except:
        return None, None

print("Processing HC-Thermo (Ureel)...")
hc_df = pd.read_csv('data/hc_thermo_combined_v7.csv')
hc_smiles = []
hc_inchikeys = []

for s in hc_df['smiles']:
    can, ikey = process_smiles(s)
    hc_smiles.append(can)
    hc_inchikeys.append(ikey)

hc_df['canonical_smiles'] = hc_smiles
hc_df['inchikey'] = hc_inchikeys

# Drop invalid molecules if any
hc_df = hc_df.dropna(subset=['inchikey'])
hc_df.to_csv('data/hc_thermo_canonical.csv', index=False)
hc_inchikeys_set = set(hc_df['inchikey'].values)
print(f"HC-Thermo valid molecules: {len(hc_df)}")

print("\nProcessing Goldsmith...")
gold_df = pd.read_csv('data/goldsmith_data/goldsmith_thermo_parsed.csv')
gold_smiles = []
gold_inchikeys = []

for s in gold_df['smiles']:
    can, ikey = process_smiles(s)
    gold_smiles.append(can)
    gold_inchikeys.append(ikey)

gold_df['canonical_smiles'] = gold_smiles
gold_df['inchikey'] = gold_inchikeys
gold_df = gold_df.dropna(subset=['inchikey'])
print(f"Goldsmith valid molecules: {len(gold_df)}")

# Deduplication
dedup_gold_df = gold_df[~gold_df['inchikey'].isin(hc_inchikeys_set)]
print(f"Goldsmith after deduplication against HC-Thermo: {len(dedup_gold_df)}")

dedup_gold_df.to_csv('data/goldsmith_data/goldsmith_deduped.csv', index=False)
print("Deduplication complete. Saved to data/goldsmith_data/goldsmith_deduped.csv and data/hc_thermo_canonical.csv")

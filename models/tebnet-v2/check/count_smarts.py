import pandas as pd
from rdkit import Chem

df = pd.read_csv('qm9.csv')
smiles_list = df['smiles'].tolist()

ether_smarts = Chem.MolFromSmarts('[OD2](C)(C)')
epoxide_smarts = Chem.MolFromSmarts('C1OC1')
enol_smarts = Chem.MolFromSmarts('[OH]-C=C')

ether_count = 0
epoxide_count = 0
enol_count = 0

print(f"Total molecules: {len(smiles_list)}")

for s in smiles_list:
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        continue
    
    if mol.HasSubstructMatch(ether_smarts):
        ether_count += 1
    
    if mol.HasSubstructMatch(epoxide_smarts):
        epoxide_count += 1
        
    if mol.HasSubstructMatch(enol_smarts):
        enol_count += 1

print(f"Ether: {ether_count}")
print(f"Epoxide: {epoxide_count}")
print(f"Enol: {enol_count}")

import re
import pandas as pd
from rdkit import Chem

def parse_rmg_adjlist_to_smiles(adjlist_str):
    """Converts RMG Adjacency List to RDKit SMILES."""
    lines = [line.strip() for line in adjlist_str.split('\n') if line.strip() and not line.startswith('multiplicity')]
    
    mol = Chem.RWMol()
    node_to_idx = {}
    
    # 1st pass: Add atoms
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        node_id = parts[0]
        element = parts[1]
        u_str = parts[2] # e.g. u0, u1
        c_str = parts[4] # e.g. c0, c+1, c-1
        
        atom = Chem.Atom(element)
        
        # Unpaired electrons (Radicals)
        if u_str.startswith('u'):
            u_val = int(u_str[1:])
            atom.SetNumRadicalElectrons(u_val)
            
        # Charge
        if c_str.startswith('c'):
            c_val_str = c_str[1:]
            if c_val_str.startswith('+'):
                atom.SetFormalCharge(int(c_val_str[1:]))
            elif c_val_str.startswith('-'):
                atom.SetFormalCharge(-int(c_val_str[1:]))
            else:
                atom.SetFormalCharge(int(c_val_str))
                
        idx = mol.AddAtom(atom)
        node_to_idx[node_id] = idx
        
    # 2nd pass: Add bonds
    for line in lines:
        parts = line.split()
        if len(parts) < 6:
            continue
        node_id = parts[0]
        idx1 = node_to_idx[node_id]
        
        bond_strs = parts[5:]
        for b_str in bond_strs:
            if not (b_str.startswith('{') and b_str.endswith('}')):
                continue
            b_content = b_str[1:-1].split(',')
            target_id = b_content[0]
            order_str = b_content[1]
            
            idx2 = node_to_idx[target_id]
            
            # Only add bond once (idx1 < idx2)
            if idx1 < idx2:
                if order_str == 'S':
                    order = Chem.BondType.SINGLE
                elif order_str == 'D':
                    order = Chem.BondType.DOUBLE
                elif order_str == 'T':
                    order = Chem.BondType.TRIPLE
                elif order_str == 'B':
                    order = Chem.BondType.AROMATIC
                else:
                    order = Chem.BondType.SINGLE
                mol.AddBond(idx1, idx2, order)
                
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except Exception as e:
        return None

def parse_rmg_library(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
        
    records = []
    
    class ThermoData:
        def __init__(self, Tdata, Cpdata, H298, S298, **kwargs):
            self.Tdata = Tdata[0] if isinstance(Tdata, tuple) else Tdata
            self.Cpdata = Cpdata[0] if isinstance(Cpdata, tuple) else Cpdata
            self.H298 = H298[0] if isinstance(H298, tuple) else H298
            self.S298 = S298[0] if isinstance(S298, tuple) else S298

    def entry(index, label, molecule, thermo, **kwargs):
        if not isinstance(thermo, ThermoData):
            return
            
        smiles = parse_rmg_adjlist_to_smiles(molecule)
        if smiles:
            # RMG uses cal/mol. TEB-Net uses J/mol.
            # 1 cal = 4.184 J
            h298_j_mol = thermo.H298 * 1000 * 4.184 # kcal/mol -> J/mol
            s298_j_mol_k = thermo.S298 * 4.184 # cal/(mol K) -> J/(mol K)
            
            cp_j_mol_k = [cp * 4.184 for cp in thermo.Cpdata]
            
            record = {
                'smiles': smiles,
                'label': label,
                'H298_J_mol': h298_j_mol,
                'S298_J_mol_K': s298_j_mol_k,
            }
            # Add Cp values
            for t, cp in zip(thermo.Tdata, cp_j_mol_k):
                record[f'Cp{t}_J_mol_K'] = cp
                
            records.append(record)
            
    exec_globals = {'entry': entry, 'ThermoData': ThermoData, 'NASA': None, 'Polynomial': None}
    try:
        exec(content, exec_globals)
    except Exception as e:
        print(f"Error executing RMG file: {e}")
        
    df = pd.DataFrame(records)
    return df

if __name__ == '__main__':
    df = parse_rmg_library('data/goldsmith_data/DFT_QCI_thermo.py')
    print(f"Parsed {len(df)} molecules successfully.")
    print(df.head())
    df.to_csv('data/goldsmith_data/goldsmith_thermo_parsed.csv', index=False)

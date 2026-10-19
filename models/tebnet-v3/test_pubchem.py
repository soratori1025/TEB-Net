import pubchempy as pcp
import requests
compounds = pcp.get_compounds('CCCCC', 'smiles')
cid = compounds[0].cid
print(f"CID: {cid}")
url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON/"
res = requests.get(url).json()
print("Success")

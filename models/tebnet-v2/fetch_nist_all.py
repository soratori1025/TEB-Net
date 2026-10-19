import pandas as pd
import urllib.request
import re
import urllib.parse
from bs4 import BeautifulSoup
import time

def get_cas_from_smiles(smiles):
    # Try Cactus CIR
    try:
        url = f"https://cactus.nci.nih.gov/chemical/structure/{urllib.parse.quote(smiles)}/cas"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        res = urllib.request.urlopen(req, timeout=5).read().decode('utf-8')
        # return the first CAS number
        for line in res.split('\n'):
            if re.match(r'\d+-\d{2}-\d', line.strip()):
                return line.strip()
    except Exception as e:
        pass
    return None

def get_nist_data(cas):
    url = f"https://webbook.nist.gov/cgi/cbook.cgi?ID=C{cas.replace('-', '')}&Units=SI&Mask=1"
    res = {'H_kcal': None, 'H_ref': None, 'S_cal': None, 'S_ref': None}
    try:
        tables = pd.read_html(url, match="gas")
        for tbl in tables:
            if 'Quantity' in tbl.columns:
                # S°gas
                s_mask = tbl['Quantity'].astype(str).str.contains("S°gas")
                if s_mask.any():
                    s_row = tbl[s_mask].iloc[0]
                    v = float(str(s_row['Value']).split('±')[0].strip())
                    res['S_cal'] = v / 4.184
                    res['S_ref'] = s_row.get('Reference', 'N/A')
                # ΔfH°gas
                h_mask = tbl['Quantity'].astype(str).str.contains("ΔfH°gas")
                if h_mask.any():
                    h_row = tbl[h_mask].iloc[0]
                    v = float(str(h_row['Value']).split('±')[0].strip())
                    # NIST gives H in kJ/mol. We want kcal/mol.
                    res['H_kcal'] = v / 4.184
                    res['H_ref'] = h_row.get('Reference', 'N/A')
    except Exception as e:
        pass
    return res

df = pd.read_csv("check/homologous_series.csv")
for idx, row in df.iterrows():
    if row['series'] in ["n-alkane", "1-alkene"] and row['n_carbon'] >= 5:
        cas = get_cas_from_smiles(row['smiles'])
        if cas:
            nist = get_nist_data(cas)
            if nist['H_kcal'] is not None:
                df.at[idx, 'ref_H_kcal'] = nist['H_kcal']
                # combine references
                refs = []
                if pd.notna(nist['H_ref']) and nist['H_ref'] != 'N/A': refs.append("H:" + str(nist['H_ref']))
                if nist['S_cal'] is not None:
                    df.at[idx, 'ref_S_cal'] = nist['S_cal']
                    if pd.notna(nist['S_ref']) and nist['S_ref'] != 'N/A': refs.append("S:" + str(nist['S_ref']))
                df.at[idx, 'ref_source'] = " | ".join(refs)
                print(f"Updated {row['series']} C{row['n_carbon']}: {nist}")
        time.sleep(0.5)

df.to_csv("check/homologous_series_updated.csv", index=False)

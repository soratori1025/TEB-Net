import pandas as pd
alkanes = {
    5: ("Pentane", "109-66-0"),
    6: ("Hexane", "110-54-3"),
    7: ("Heptane", "142-82-5"),
    8: ("Octane", "111-65-9"),
    9: ("Nonane", "111-84-2"),
    10: ("Decane", "124-18-5"),
    11: ("Undecane", "1120-21-4"),
    12: ("Dodecane", "112-40-3"),
    13: ("Tridecane", "629-50-5"),
    14: ("Tetradecane", "629-59-4"),
    15: ("Pentadecane", "629-62-9"),
    16: ("Hexadecane", "544-76-3"),
}
for n, (name, cas) in alkanes.items():
    url = f"https://webbook.nist.gov/cgi/cbook.cgi?ID=C{cas.replace('-', '')}&Units=SI&Mask=1"
    try:
        tables = pd.read_html(url, match="gas")
        for tbl in tables:
            if 'Quantity' in tbl.columns:
                gas_entropy = tbl[tbl['Quantity'] == 'S°gas']
                if not gas_entropy.empty:
                    val = float(str(gas_entropy['Value'].iloc[0]).split('±')[0].strip())
                    ref = gas_entropy['Reference'].iloc[0] if 'Reference' in gas_entropy.columns else "N/A"
                    val_cal = val / 4.184
                    print(f"| {n} | {name} | {val_cal:.2f} cal/mol*K | Ref: {ref} |")
                    break
    except Exception as e:
        print(f"Error for {n} {name}")

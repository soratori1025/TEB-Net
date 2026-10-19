import pandas as pd
url = "https://webbook.nist.gov/cgi/cbook.cgi?ID=C109671&Units=SI&Mask=1"
tables = pd.read_html(url, match="gas")
for tbl in tables:
    if 'Quantity' in tbl.columns:
        print(tbl)

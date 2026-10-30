import urllib.request
from bs4 import BeautifulSoup
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
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    html = urllib.request.urlopen(req).read()
    soup = BeautifulSoup(html, 'html.parser')
    val = None
    ref = None
    for tr in soup.find_all('tr'):
        tds = tr.find_all('td')
        if len(tds) > 1 and "S°gas" in tds[0].get_text():
            try:
                text = tds[1].get_text()
                v = float(text.replace('±', '').split()[0])
                val = v / 4.184
                if len(tds) > 4:
                    ref = tds[4].get_text(strip=True)
                break
            except Exception:
                pass
    if val is not None:
        print(f"| {n} | {name} | {val:.2f} cal/mol*K | Ref: {ref} |")
    else:
        print(f"| {n} | {name} | N/A |")

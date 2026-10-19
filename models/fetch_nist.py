import urllib.request
import re

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

print("| n | Alkane | CAS | S° gas (J/mol*K) | S° gas (cal/mol*K) |")
print("|---|---|---|---|---|")

for n, (name, cas) in alkanes.items():
    url = f"https://webbook.nist.gov/cgi/cbook.cgi?ID=C{cas.replace('-', '')}&Units=SI&Mask=1"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            html = response.read().decode('utf-8')
        m = re.search(r'S&deg;<sub>gas</sub>(?:.*?<td class="right-nowrap">)([\d\.]+)(?:\s*&plusmn;\s*[\d\.]+)?(?:</td>)', html)
        if m:
            s_j = float(m.group(1))
            s_cal = s_j / 4.184
            print(f"| {n} | {name} | {cas} | {s_j:.1f} | {s_cal:.2f} |")
        else:
            print(f"| {n} | {name} | {cas} | N/A | N/A |")
    except Exception as e:
        print(f"| {n} | {name} | {cas} | Error: {e} | |")


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

for n, (name, cas) in alkanes.items():
    url = f"https://webbook.nist.gov/cgi/cbook.cgi?ID=C{cas.replace('-', '')}&Units=SI&Mask=1"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        html = urllib.request.urlopen(req).read().decode('utf-8')
        # Find lines with S&deg;<sub>gas
        for line in html.split('</tr>'):
            if 'S&deg;<sub>gas</sub>' in line and '<td class="symbol">' not in line:
                m1 = re.search(r'<td class="right-nowrap">([\d\.]+)', line)
                m2 = re.search(r'<a href="#ref[^>]*>([^<]+)</a>', line)
                if m1:
                    val = float(m1.group(1)) / 4.184
                    ref = m2.group(1) if m2 else "N/A"
                    print(f"| {n} | {name} | {val:.2f} cal/mol*K | Ref: {ref} |")
                break
    except Exception as e:
        print(f"Error for {n} {name}")


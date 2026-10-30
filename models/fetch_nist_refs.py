import urllib.request
import re
alkanes = {
    5: ("Pentane", "109-66-0"),
    6: ("Hexane", "110-54-3"),
    7: ("Heptane", "142-82-5"),
    8: ("Octane", "111-65-9"),
    16: ("Hexadecane", "544-76-3"),
}
for n, (name, cas) in alkanes.items():
    url = f"https://webbook.nist.gov/cgi/cbook.cgi?ID=C{cas.replace('-', '')}&Units=SI&Mask=1"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            html = response.read().decode('utf-8')
        print(f"\n--- {name} ---")
        matches = re.finditer(r'S&deg;<sub>gas</sub>(?:.*?<td class="right-nowrap">)([\d\.]+)(?:\s*&plusmn;\s*[\d\.]+)?(?:</td>).*?(?:<td class="left">)(.*?)(?:</td>)', html, re.DOTALL)
        for m in matches:
            val = float(m.group(1)) / 4.184
            ref_html = m.group(2)
            ref_text = re.sub(r'<[^>]+>', '', ref_html).strip()
            print(f"  {val:.2f} cal/mol*K | Ref: {ref_text}")
    except Exception as e:
        print(f"Error for {name}: {e}")

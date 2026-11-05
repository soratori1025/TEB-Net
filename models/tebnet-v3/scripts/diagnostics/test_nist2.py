import urllib.request
import re
url = "https://webbook.nist.gov/cgi/cbook.cgi?ID=C109671&Units=SI&Mask=1"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
res = urllib.request.urlopen(req, timeout=5).read().decode('utf-8')
print("delta H gas" in res, "S gas" in res)
matches = re.findall(r'fH&deg;gas.*?<td[^>]*>(-?\d+\.?\d*)', res, re.DOTALL)
print("H:", matches)

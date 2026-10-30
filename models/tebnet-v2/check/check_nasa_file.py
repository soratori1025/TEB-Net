"""Physical checks of any Chemkin/NASA-7 thermo file up to its upper temperature (no model needed).

For every species: Cp(T) on 300..T_high against the classical limit of an ideal gas, monotonicity of Cp, and
whether Cp turns negative. The limit used is the lenient (3N - 3/2)R (linear molecule), which no real
polyatomic molecule exceeds by much; N is the atom count from the element fields of the first line.
Also reports how many species use one coefficient set for both temperature ranges.

Usage
  python check_nasa_file.py deeptherm_nasa.dat [--t_max 3000]
"""
from __future__ import annotations
import argparse
import re
import numpy as np
R = 8.314462618
NUM = re.compile(r"[+-]?\d\.\d+[Ee][+-]\d+")
def parse(path):
    lines = open(path).read().splitlines()
    out, i = [], 0
    while i + 3 < len(lines):
        l = lines[i]
        if len(l) >= 80 and l[79] == "1":
            c = [float(x) for x in NUM.findall(lines[i + 1][:75] + lines[i + 2][:75] + lines[i + 3][:60])]
            el = {e.capitalize(): int(n) for e, n in re.findall(r"([A-Za-z]{1,2})\s*(\d+)", l[24:44]) if int(n) > 0}
            try:
                t_lo, t_hi, t_mid = float(l[45:55]), float(l[55:65]), float(l[65:73])
            except ValueError:
                i += 1
                continue
            if len(c) == 14:
                out.append((l[:18].strip(), el, t_lo, t_hi, t_mid, np.array(c[:7]), np.array(c[7:])))
            i += 4
        else:
            i += 1
    return out
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("thermo_file")
    ap.add_argument("--t_max", type=float, default=None, help="default: each species' own upper temperature")
    args = ap.parse_args()
    sp = parse(args.thermo_file)
    if not sp:
        raise SystemExit("no NASA-7 entries found")
    n = len(sp)
    same = sum(np.allclose(hi, lo) for *_, hi, lo in sp)
    over = over1 = dec = neg = 0
    first, ratio = [], []
    for name, el, t_lo, t_hi, t_mid, hi, lo in sp:
        T = np.arange(300.0, (args.t_max or t_hi) + 1.0, 10.0)
        a = np.where((T <= t_mid)[:, None], lo[None, :], hi[None, :])
        cp = R * (a[:, 0] + a[:, 1] * T + a[:, 2] * T**2 + a[:, 3] * T**3 + a[:, 4] * T**4)
        N = sum(el.values())
        lim = 2.5 * R if N == 1 else (3 * N - 1.5) * R
        o = cp > lim
        over += o.any()
        over1 += (cp > lim + R).any()
        dec += (np.diff(cp) < -1e-6).any()
        neg += (cp < 0).any()
        if o.any():
            first.append(T[np.argmax(o)])
        ratio.append(cp[-1] / lim)
    pct = lambda k: f"{k} ({100 * k / n:.1f} %)"
    print(f"{args.thermo_file}: {n} species; one coefficient set for both ranges: {pct(same)}")
    print(f"| Check (300 K to {'T_high' if not args.t_max else f'{args.t_max:g} K'}) | Species |")
    print("|---|---|")
    print(f"| Cp above the classical limit (3N - 3/2)R | {pct(over)} |")
    print(f"| ... by more than 1 R | {pct(over1)} |")
    print(f"| median temperature of the first crossing | {np.median(first):.0f} K |" if first else "| median temperature of the first crossing | – |")
    print(f"| Cp decreasing with T somewhere | {pct(dec)} |")
    print(f"| Cp negative somewhere | {pct(neg)} |")
    r = np.percentile(ratio, [5, 50, 95])
    print(f"| Cp(T_max) / limit, 5th / 50th / 95th percentile | {r[0]:.2f} / {r[1]:.2f} / {r[2]:.2f} |")
if __name__ == "__main__":
    main()

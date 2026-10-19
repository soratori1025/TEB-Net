"""High-temperature Cp references for Figure 4: C/H/O species with NASA polynomials to >= 3000 K in RMG-database.

The DeepTherm data stop at 1500 K, so every species with a trustworthy Cp(T) up to 3000 K tests the temperature
extrapolation of TEB-Net's exported polynomials, whether or not the species itself is in DeepTherm. This script
collects such species from the thermo libraries of RMG-database and writes one CSV row per (species, library).

Two kinds of source (column `source`):
  qm         statistical mechanics on quantum-chemical frequencies (rigid rotor - harmonic oscillator, with 1D hindered
             rotors where the library says so). For the Arkane / ARC libraries, Arkane computes Cp from the partition
             function at 10-3000 K and fits it (Wilhoit form, then NASA-7), so the reference values up to 3000 K are
             computed, not extrapolated. Arkane fixes the Wilhoit Cp_inf at (3N - 2 - n_rotors/2) R, the same
             expected limit TEB-Net's completion uses (with its own rotor count); at 3000 K Cp is still below that
             limit and the curve between 1500 and 3000 K is set by the vibrational frequencies, so the comparison
             tests the path, not the limit. Anharmonicity is not included.
  evaluated  Burcat / ATcT / JANAF-type polynomials of small species in FFCM-1, GRI-Mech 3.0, the Hashemi,
             Klippenstein, Glarborg et al. (2016) model and RMG's primary library (anharmonic corrections included
             for many of them); independent of the convention above.
Mechanism libraries whose polynomials are group-additivity estimates (JetSurF, USC Mech II, ...) are not used, and
ThermoData entries (tabulated to 1500 K and extrapolated by RMG) are skipped.

The library files are Python source; they are read with `ast` (nothing is executed). RDKit turns the adjacency
lists into canonical SMILES without stereochemistry (DeepTherm SMILES are canonicalised the same way before any
match); without RDKit
the SMILES column stays empty and the CSV cannot be matched to TEB-Net predictions.

Usage
  git clone --depth 1 --filter=blob:none --sparse https://github.com/ReactionMechanismGenerator/RMG-database.git
  git -C RMG-database sparse-checkout set input/thermo/libraries
  python check/rmg_high_t_reference.py --rmg_db RMG-database --deeptherm_csv models-baseline/deeptherm_data.csv \
      --out rmg_cp_reference.csv
Then run check/nasa7_check.py --smiles_csv rmg_cp_reference.csv (unique SMILES) and check/high_t_validation.py.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

R = 8.314462618
T_REF = 298.15
T_GRID = np.arange(300.0, 3000.0 + 1.0, 100.0)

# priority order: the first library that has a species is the one used (rank 0); evaluated data before QM,
# higher-level QM before CBS-QB3 RRHO
LIBRARIES = [
    ("FFCM1(-)", "evaluated"),
    ("GRI-Mech3.0", "evaluated"),
    ("primaryThermoLibrary", "evaluated"),
    ("Klippenstein_Glarborg2016", "evaluated"),    # Burcat / ATcT polynomials of the Hashemi et al. (2016) model
    ("heavy_oil_ccsdtf12_1dHR", "qm"),
    ("Elliott_OOQOOH", "qm"),
    ("Spiekermann_refining_elementary_reactions", "qm"),
    ("FormicAcid", "qm"),
    ("SABIC_aromatics_1dHR", "qm"),
    ("SABIC_aromatics_1dHR_extended", "qm"),
    ("Butadiene_Dimerization", "qm"),
    ("s3_5_7_ane", "qm"),
    ("2-BTP_G4", "qm"),
    ("SABIC_aromatics", "qm"),              # CBS-QB3, RRHO without hindered rotors: lowest priority
]
ELEMENTS = {"H", "He", "C", "N", "O", "F", "Ne", "Si", "P", "S", "Cl", "Ar", "Br", "I", "Li"}
ALLOWED = {"C", "H", "O"}


# ───────────────────────── safe reader of RMG library files ─────────────────────────
class Call(dict):
    """A call such as NASA(...) in a library file: keyword arguments, positional ones in 'args', name in .fn."""
    def __init__(self, fn, args, kw):
        super().__init__(kw)
        self.fn = fn
        self["args"] = args


def _eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_eval(e) for e in node.elts]
    if isinstance(node, ast.Dict):
        return {_eval(k): _eval(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _eval(node.operand)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        a, b = _eval(node.left), _eval(node.right)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return {ast.Add: a + b, ast.Sub: a - b, ast.Mult: a * b, ast.Div: a / b if b else np.nan}[type(node.op)]
        if isinstance(a, str) and isinstance(b, str) and isinstance(node.op, ast.Add):
            return a + b
        return None
    if isinstance(node, ast.Call):
        fn = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "?")
        return Call(fn, [_eval(a) for a in node.args], {k.arg: _eval(k.value) for k in node.keywords if k.arg})
    if isinstance(node, ast.Name):
        return {"True": True, "False": False, "None": None}.get(node.id, node.id)
    return None


def read_library(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8", errors="replace"), filename=str(path))
    for st in tree.body:
        if isinstance(st, ast.Expr) and isinstance(st.value, ast.Call) and getattr(st.value.func, "id", "") == "entry":
            yield _eval(st.value)


# ───────────────────────── adjacency list ─────────────────────────
def parse_adjlist(text):
    """RMG adjacency list -> (atoms [(element, radicals, charge)], bonds {(i, j): order}, multiplicity or None)."""
    atoms, bonds, mult = [], {}, None
    for raw in text.strip().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.lower().startswith("multiplicity"):
            mult = int(line.split()[1])
            continue
        tok = line.split()
        idx = int(tok[0])
        k = 1
        if tok[k].startswith("*"):                          # optional atom label
            k += 1
        el = tok[k]
        if el not in ELEMENTS:
            raise ValueError(f"atom type {el!r} is not an element")
        u = c = 0
        for t in tok[k + 1:]:
            if t.startswith("{"):
                j, order = t.strip("{}").split(",")
                key = tuple(sorted((idx, int(j))))
                bonds[key] = order
            elif t.isdigit():                               # old format: radical count without 'u'
                u = int(t)
            elif t[0] == "u":
                u = int(t[1:])
            elif t[0] == "c":
                c = int(t[1:])
            elif t[0] == "p":
                pass
            else:
                raise ValueError(f"cannot read {t!r}")
        if idx != len(atoms) + 1:
            raise ValueError("atoms are not numbered 1..N")
        atoms.append((el, u, c))
    return atoms, bonds, mult


def to_smiles(atoms, bonds):
    from rdkit import Chem
    order = {"S": Chem.BondType.SINGLE, "D": Chem.BondType.DOUBLE, "T": Chem.BondType.TRIPLE,
             "Q": Chem.BondType.QUADRUPLE, "B": Chem.BondType.AROMATIC, "1": Chem.BondType.SINGLE,
             "2": Chem.BondType.DOUBLE, "3": Chem.BondType.TRIPLE, "1.5": Chem.BondType.AROMATIC}
    rw = Chem.RWMol()
    for el, u, c in atoms:
        a = Chem.Atom(el)
        a.SetFormalCharge(c)
        a.SetNumRadicalElectrons(u)
        a.SetNoImplicit(True)                               # RMG lists every H explicitly
        rw.AddAtom(a)
    for (i, j), o in bonds.items():
        bt = order[o]
        rw.AddBond(i - 1, j - 1, bt)
        if bt == Chem.BondType.AROMATIC:
            b = rw.GetBondBetweenAtoms(i - 1, j - 1)
            b.SetIsAromatic(True)
            rw.GetAtomWithIdx(i - 1).SetIsAromatic(True)
            rw.GetAtomWithIdx(j - 1).SetIsAromatic(True)
    m = rw.GetMol()
    Chem.SanitizeMol(m)
    m = Chem.RemoveHs(m)
    return Chem.MolToSmiles(m, isomericSmiles=False)


def roundtrip_ok(smiles, n_rad, n_h):
    """False if the SMILES does not carry the adjacency list's radical electrons / H count (e.g. singlet
    carbenes: '[CH2]' is read back as a triplet), so that TEB-Net would see another species."""
    from rdkit import Chem
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return False
    return (sum(a.GetNumRadicalElectrons() for a in m.GetAtoms()) == n_rad
            and sum(a.GetTotalNumHs() for a in m.GetAtoms()) == n_h)


def canonical(smiles):
    from rdkit import Chem
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(m, isomericSmiles=False) if m is not None else None


# ───────────────────────── NASA polynomials ─────────────────────────
def _kelvin(q):
    if isinstance(q, (int, float)):
        return float(q)
    v, unit = q[0], (q[1] if len(q) > 1 else "K")
    if unit != "K":
        raise ValueError(f"temperature unit {unit!r}")
    return float(v)


def nasa_pieces(thermo):
    """NASA(...) -> [(t_min, t_max, coeffs)] sorted by t_min (7 or 9 coefficients)."""
    out = []
    for p in thermo.get("polynomials") or []:
        if not isinstance(p, Call):
            continue
        coeffs = p.get("coeffs") or (p["args"][0] if p["args"] else None)
        if coeffs is None or len(coeffs) not in (7, 9):
            raise ValueError("NASA polynomial without 7 or 9 coefficients")
        out.append((_kelvin(p["Tmin"]), _kelvin(p["Tmax"]), [float(x) for x in coeffs]))
    return sorted(out)


def _piece(pieces, T):
    for lo, hi, a in pieces:
        if lo - 1e-6 <= T <= hi + 1e-6:
            return a
    return None


def cp_j(pieces, T):
    a = _piece(pieces, T)
    if a is None:
        return np.nan
    if len(a) == 7:
        return R * sum(a[k] * T ** k for k in range(5))
    return R * (a[0] / T ** 2 + a[1] / T + a[2] + a[3] * T + a[4] * T ** 2 + a[5] * T ** 3 + a[6] * T ** 4)


def h_j(pieces, T):
    a = _piece(pieces, T)
    if a is None:
        return np.nan
    if len(a) == 7:
        return R * T * (sum(a[k] * T ** k / (k + 1) for k in range(5)) + a[5] / T)
    return R * T * (-a[0] / T ** 2 + a[1] * np.log(T) / T + a[2] + a[3] * T / 2 + a[4] * T ** 2 / 3
                    + a[5] * T ** 3 / 4 + a[6] * T ** 4 / 5 + a[7] / T)


def s_j(pieces, T):
    a = _piece(pieces, T)
    if a is None:
        return np.nan
    if len(a) == 7:
        return R * (a[0] * np.log(T) + a[1] * T + a[2] * T ** 2 / 2 + a[3] * T ** 3 / 3 + a[4] * T ** 4 / 4 + a[6])
    return R * (-a[0] / (2 * T ** 2) - a[1] / T + a[2] * np.log(T) + a[3] * T + a[4] * T ** 2 / 2
                + a[5] * T ** 3 / 3 + a[6] * T ** 4 / 4 + a[8])


def covers(pieces, t_lo, t_hi):
    """True if the pieces cover [t_lo, t_hi] without gaps."""
    t = t_lo
    for lo, hi, _ in pieces:
        if lo <= t + 1e-6 and hi > t:
            t = hi
    return t >= t_hi - 1e-6


# ───────────────────────── main ─────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rmg_db", required=True, help="RMG-database checkout (or its input/thermo/libraries folder)")
    ap.add_argument("--libraries", nargs="*", default=None, help="default: the list in this script, in priority order")
    ap.add_argument("--t_need", type=float, default=3000.0, help="polynomials must cover 298.15 K .. t_need")
    ap.add_argument("--deeptherm_csv", default=None, help="adds in_deeptherm (canonical SMILES match)")
    ap.add_argument("--out", default="rmg_cp_reference.csv")
    args = ap.parse_args()

    root = Path(args.rmg_db)
    libdir = root if root.name == "libraries" else root / "input" / "thermo" / "libraries"
    libs = [(l, dict(LIBRARIES).get(l, "qm")) for l in args.libraries] if args.libraries else LIBRARIES
    try:
        import rdkit  # noqa: F401
        have_rdkit = True
    except ImportError:
        have_rdkit = False
        print("WARNING: RDKit not found - the smiles column stays empty and cannot be matched to TEB-Net.")

    rows, stats = [], []
    for prio, (lib, source) in enumerate(libs):
        path = libdir / f"{lib}.py"
        if not path.exists():
            print(f"  {lib}: not found, skipped")
            continue
        n = {"entries": 0, "NASA": 0, "C/H/O": 0, f"to {args.t_need:g} K": 0, "used": 0}
        for e in read_library(path):
            n["entries"] += 1
            th = e.get("thermo")
            if not isinstance(th, Call) or th.fn != "NASA":
                continue
            n["NASA"] += 1
            mol = e.get("molecule")
            if not isinstance(mol, str):
                continue
            try:
                atoms, bonds, mult = parse_adjlist(mol) if "\n" in mol.strip() or "{" in mol else (None, None, None)
            except (ValueError, IndexError):
                continue
            if atoms is None:                                # a SMILES string instead of an adjacency list
                if not have_rdkit:
                    continue
                smi = canonical(mol)
                if smi is None:
                    continue
                from rdkit import Chem
                mh = Chem.AddHs(Chem.MolFromSmiles(smi))
                atoms = [(a.GetSymbol(), a.GetNumRadicalElectrons(), a.GetFormalCharge()) for a in mh.GetAtoms()]
            els = [a[0] for a in atoms]
            if not set(els) <= ALLOWED or "C" not in els or any(a[2] != 0 for a in atoms) or len(atoms) < 2:
                continue
            n["C/H/O"] += 1
            try:
                pieces = nasa_pieces(th)
            except (ValueError, TypeError, KeyError):
                continue
            if not pieces or not covers(pieces, T_REF, args.t_need):
                continue
            n[f"to {args.t_need:g} K"] += 1
            smi = ""
            if have_rdkit and bonds is not None:
                try:
                    smi = to_smiles(atoms, bonds)
                except Exception as ex:                      # noqa: BLE001 (RDKit raises several types)
                    print(f"  {lib}: {e.get('label')} rejected by RDKit ({type(ex).__name__})")
                    continue
            elif have_rdkit:
                smi = canonical(mol)
            cnt = {el: els.count(el) for el in ("C", "H", "O") if el in els}
            n_rad = sum(a[1] for a in atoms)
            cpinf = th.get("CpInf")
            row = {"smiles": smi, "library": lib, "source": source, "priority": prio, "label": e.get("label"),
                   "formula": "".join(f"{k}{v if v > 1 else ''}" for k, v in cnt.items()),
                   "n_atoms": len(atoms), "n_heavy": len(atoms) - cnt.get("H", 0),
                   "multiplicity": int(mult) if mult else n_rad + 1, "n_radical_electrons": n_rad,
                   "smiles_roundtrip": int(roundtrip_ok(smi, n_rad, cnt.get("H", 0))) if smi else 0,
                   "t_min": pieces[0][0], "t_max": max(p[1] for p in pieces),
                   "cp_inf_ref": float(cpinf[0]) if isinstance(cpinf, list) and cpinf and cpinf[1:] in
                   ([], ["J/(mol*K)"]) else np.nan,
                   "h298_kJ": h_j(pieces, T_REF) / 1000.0, "s298_J": s_j(pieces, T_REF),
                   "nasa": json.dumps([[lo, hi, a] for lo, hi, a in pieces])}
            for T in T_GRID:
                row[f"cp_{T:.0f}"] = cp_j(pieces, T)
            rows.append(row)
            n["used"] += 1
        stats.append((lib, source, n))

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no usable entries")
    key = df.smiles.where(df.smiles != "", df.library + ":" + df.label.astype(str)) + "|" + df.multiplicity.astype(str)
    df["rank"] = df.assign(_k=key).sort_values(["priority"], kind="stable").groupby("_k").cumcount().reindex(df.index)
    if args.deeptherm_csv and have_rdkit:
        d = pd.read_csv(args.deeptherm_csv)
        col = next(c for c in d.columns if c.strip().lower() == "smiles")
        dt = {canonical(str(s).strip()) for s in d[col]}
        df["in_deeptherm"] = df.smiles.isin(dt).astype(int)
    df = df.sort_values(["rank", "priority", "n_atoms"]).drop(columns="priority")
    df.to_csv(args.out, index=False, float_format="%.6g")

    print(f"\n| Library | Source | Entries | NASA | C/H/O, neutral, with C | covering 298–{args.t_need:g} K |")
    print("|---|---|---|---|---|---|")
    for lib, source, n in stats:
        print(f"| {lib} | {source} | {n['entries']} | {n['NASA']} | {n['C/H/O']} | {n['used']} |")
    u = df[df["rank"] == 0]
    print(f"\n{len(df)} rows, {len(u)} unique species ({(u.source == 'qm').sum()} qm, "
          f"{(u.source == 'evaluated').sum()} evaluated); heavy atoms {u.n_heavy.min()}–{u.n_heavy.max()} "
          f"(median {u.n_heavy.median():.0f}); multiplicity > 2: {(u.multiplicity > 2).sum()}; "
          f"SMILES does not round-trip (e.g. singlet carbenes): {(u.smiles_roundtrip == 0).sum()}")
    if "in_deeptherm" in u:
        print(f"in DeepTherm: {int(u.in_deeptherm.sum())} (their Cp <= 1500 K can be compared with the data)")
    ratio = u.cp_3000 / u.cp_inf_ref
    if ratio.notna().any():
        print(f"reference Cp(3000 K) / its own Cp_inf: median {ratio.median():.3f} "
              f"(5th–95th pct {ratio.quantile(0.05):.3f}–{ratio.quantile(0.95):.3f})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
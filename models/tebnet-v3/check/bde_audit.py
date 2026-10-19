"""Label-consistency audit of a thermochemistry dataset (no model involved).

1. BDE audit. For every mono-radical R• whose closed-shell parent R–H is also in the dataset:
       BDE(R–H) = ΔHf(R•) + ΔHf(H•) − ΔHf(RH),    ΔHf(H•) = 52.10 kcal/mol (ATcT)
   BDEs of a given C–H / O–H class are narrow (a few kcal/mol). An entry far from its class
   median points to a label or SMILES problem, independently of any ML model.

2. Attribution. A BDE involves two labels, so an outlier alone does not say which one is wrong.
   The siblings (other radicals of the same parent) decide:
     parent     ≥ 2 siblings of ≥ 2 DIFFERENT classes all deviate in the same direction
                → the shared parent ΔHf is off (estimated error = −median deviation)
     radical    the outlier has at least one normal sibling → the parent is fine, the radical is off
     ambiguous  no normal sibling of another class (single radical, or all siblings of one class,
                e.g. 1-heptene whose only radicals in the set are two vinyl radicals)
   "est_error_kcal" is the estimated (label − expected) of the suspected entry.

3. Duplicates. Resonance structures of one radical (C=C[O] / [CH2]C=O) and stereo / non-stereo
   SMILES of one species get different canonical SMILES, so a dataset can hold the same species
   twice with different labels. Species with the same heavy-atom skeleton, the same H count on every
   atom and the same number of unpaired electrons are grouped.

Audit classes. radical_center_type (the model's featurizer, unchanged) plus finer classes used
only here: acyloxy RC(O)O•, hydroxyl •OH, acylperoxy RC(O)OO•, cyclopropyl, pentadienyl
(radical conjugated with two C=C / C≡C), heteroaryl (radical on an O-containing aromatic ring).
"Typical" values are for orientation (Blanksby & Ellison, Acc. Chem. Res. 36 (2003) 255 and
standard values); the attribution uses the data's own class medians.

  python scripts/bde_audit.py --csv deeptherm_data.csv
  -> bde_audit_all.csv, bde_audit_flagged.csv, bde_audit_suspects.csv, bde_audit_duplicates.csv
"""
from __future__ import annotations
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from rdkit import Chem  # noqa: E402
from rdkit import RDLogger  # noqa: E402
from rdkit.Chem import rdMolDescriptors  # noqa: E402
from tebnet.data.featurizer import radical_center_type  # noqa: E402

RDLogger.DisableLog("rdApp.*")
HF_H_ATOM = 52.10  # kcal/mol
BT = Chem.BondType

TYPICAL = {            # kcal/mol, representative R–H bonds (orientation only)
    "rad_alkyl_p": 101.1,     # C2H5–H (CH3–H is 105.0)
    "rad_alkyl_s": 98.6,      # i-C3H7–H
    "rad_alkyl_t": 96.5,      # t-C4H9–H
    "rad_alpha_O": 96.1,      # H–CH2OH
    "rad_alpha_CO": 96.0,     # H–CH2C(O)CH3
    "rad_allylic": 88.8,      # allyl–H (benzyl 89.7)
    "rad_pentadienyl": 76.5,  # 1,4-pentadiene C3–H, 1,4-cyclohexadiene C–H ≈ 76–77
    "rad_vinyl": 110.7,       # vinyl–H
    "rad_aryl": 112.9,        # phenyl–H
    "rad_heteroaryl": 120.0,  # furyl C–H ≈ 120
    "rad_cyclopropyl": 106.3, # c-C3H5–H
    "rad_acyl": 89.4,         # CH3C(O)–H
    "rad_alkoxy": 104.6,      # CH3O–H
    "rad_hydroxyl": 118.8,    # HO–H
    "rad_acyloxy": 111.0,     # HC(O)O–H, CH3C(O)O–H ≈ 110–112
    "rad_peroxy": 87.8,       # HOO–H (ROO–H ≈ 85–86)
    "rad_ethynyl": 133.3,     # HC≡C–H
    "rad_vinoxy": 86.0,       # vinyl alcohol / phenol O–H ≈ 85–87 (resonance-stabilised)
    # rad_captodative, rad_acylperoxy: no single reference value -> robust z-score only
}


# ───────────────────────── audit classes ─────────────────────────
def _is_carbonyl_c(n) -> bool:
    return n.GetAtomicNum() == 6 and any(
        b.GetBondType() == BT.DOUBLE and b.GetOtherAtom(n).GetAtomicNum() == 8 for b in n.GetBonds())


def _n_conjugated_pi(a) -> int:
    """C=C / C≡C bonds conjugated with a radical carbon: R•–C=C counts 1, R•–C=C–C=C counts 2.
    Aromatic bonds are not counted (benzyl ≈ allyl, not pentadienyl)."""
    pi = (BT.DOUBLE, BT.TRIPLE)
    seen, count = set(), 0
    for n in a.GetNeighbors():
        if n.GetAtomicNum() != 6:
            continue
        for b in n.GetBonds():
            m = b.GetOtherAtom(n)
            if b.GetBondType() not in pi or m.GetIdx() == a.GetIdx() or m.GetAtomicNum() != 6:
                continue
            if b.GetIdx() not in seen:
                seen.add(b.GetIdx())
                count += 1
            for b2 in m.GetBonds():   # extended conjugation R•–C=C–C=C
                k = b2.GetOtherAtom(m)
                if k.GetIdx() in (n.GetIdx(), a.GetIdx()) or k.GetAtomicNum() != 6:
                    continue
                for b3 in k.GetBonds():
                    if (b3.GetBondType() in pi and b3.GetIdx() not in seen
                            and b3.GetOtherAtom(k).GetAtomicNum() == 6 and b3.GetOtherAtom(k).GetIdx() != m.GetIdx()):
                        seen.add(b3.GetIdx())
                        count += 1
    return count


def audit_type(a) -> str | None:
    t = radical_center_type(a)
    if t is None:
        return None
    heavy = [n for n in a.GetNeighbors() if n.GetAtomicNum() > 1]
    if a.GetAtomicNum() == 8:
        if not heavy:
            return "rad_hydroxyl"
        if t == "rad_vinoxy" and _is_carbonyl_c(heavy[0]):
            return "rad_acyloxy"
        if t == "rad_peroxy":
            o2 = next(n for n in heavy if n.GetAtomicNum() == 8)
            if any(_is_carbonyl_c(n) for n in o2.GetNeighbors() if n.GetIdx() != a.GetIdx()):
                return "rad_acylperoxy"
        return t
    if t == "rad_aryl":
        ri = a.GetOwningMol().GetRingInfo()
        mol = a.GetOwningMol()
        for ring in ri.AtomRings():
            if a.GetIdx() in ring and any(mol.GetAtomWithIdx(j).GetAtomicNum() == 8 for j in ring):
                return "rad_heteroaryl"
        return t
    if t in ("rad_alkyl_s", "rad_alkyl_t", "rad_alpha_O", "rad_alpha_CO") and a.IsInRingSize(3):
        return "rad_cyclopropyl"
    if t == "rad_allylic" and _n_conjugated_pi(a) >= 2:
        return "rad_pentadienyl"
    return t


# ───────────────────────── helpers ─────────────────────────
def canon(smi: str) -> str | None:
    m = Chem.MolFromSmiles(str(smi))
    return Chem.MolToSmiles(m) if m is not None else None


def parent_of(mol: Chem.Mol) -> tuple[str | None, str | None, str | None]:
    rad = [a for a in mol.GetAtoms() if a.GetNumRadicalElectrons() > 0]
    if len(rad) != 1 or rad[0].GetNumRadicalElectrons() != 1:
        return None, None, None
    a = rad[0]
    typ, atyp = radical_center_type(a), audit_type(a)
    n_h = a.GetTotalNumHs()
    rw = Chem.RWMol(mol)
    b = rw.GetAtomWithIdx(a.GetIdx())
    b.SetNumRadicalElectrons(0)
    b.SetNoImplicit(True)
    b.SetNumExplicitHs(n_h + 1)
    try:
        Chem.SanitizeMol(rw)
    except Exception:
        return None, typ, atyp
    return Chem.MolToSmiles(rw), typ, atyp


def skeleton_key(mol: Chem.Mol) -> str | None:
    """Heavy-atom skeleton + H count per atom + unpaired electrons; bond orders, aromaticity and
    stereo dropped. Equal keys = resonance structures / stereo variants of one species.

    The hydrogens are made explicit ATOMS before the bonds are flattened, so the canonical SMILES
    carries them as [H] nodes. (The first version stored them as H counts on the heavy atoms; the
    SMILES writer then dropped them for under-valent organic-subset atoms and CCCO / CC#CO got the
    same key.) Formula and the sorted per-atom H pattern are part of the key as a second guard."""
    try:
        n_rad = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
        formula = rdMolDescriptors.CalcMolFormula(mol)
        h_pattern = ".".join(sorted(f"{a.GetSymbol()}{a.GetTotalNumHs()}" for a in mol.GetAtoms()))
        rw = Chem.RWMol(Chem.AddHs(mol))
        for a in rw.GetAtoms():
            a.SetNoImplicit(True)
            a.SetNumExplicitHs(0)
            a.SetNumRadicalElectrons(0)
            a.SetIsAromatic(False)
            a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        for b in rw.GetBonds():
            b.SetBondType(BT.SINGLE)
            b.SetIsAromatic(False)
            b.SetStereo(Chem.BondStereo.STEREONONE)
        m = rw.GetMol()
        m.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(m)
        smi = Chem.MolToSmiles(m, isomericSmiles=False, canonical=True)
        return f"{formula}|{h_pattern}|{smi}|rad{n_rad}"
    except Exception:
        return None


def attribute(res: pd.DataFrame, tol: float, tol_sib: float, zmax: float = 4.0) -> list[dict]:
    """Sibling-based attribution of BDE outliers to the parent or the radical label."""
    out = []
    for parent, g in res.groupby("parent"):
        dev = g.dev_median.to_numpy()
        big = np.abs(dev) > tol_sib
        sgn = np.sign(dev)
        prow, hp = int(g.parent_row.iloc[0]), float(g.Hf_parent.iloc[0])
        sib_txt = "; ".join(f"{s} [{t[4:]}] {d:+.1f}" for s, t, d in zip(g.smiles, g.audit_type, dev))
        shift = 0.0
        if len(g) >= 2 and big.all() and (sgn == sgn[0]).all() and g.audit_type.nunique() >= 2:
            shift = float(np.median(dev))
            out.append({"row": prow, "smiles": parent, "role": "parent", "est_error_kcal": round(-shift, 1),
                        "Hf_label": hp, "evidence": f"{len(g)} radicals of {g.audit_type.nunique()} classes "
                                                    f"shifted alike: {sib_txt}"})
        for (_, r), d in zip(g.iterrows(), dev - shift):
            mad = r.dev_median / r.robust_z if r.robust_z != 0 else 1.0
            if not (abs(d) > tol or (abs(d) > tol_sib and abs(d / mad) > zmax)):
                continue
            others = g[g.row != r.row]
            normal_other = others[(others.dev_median.abs() <= tol_sib)]
            if shift != 0.0 or len(normal_other) > 0:
                role = "radical"
                ev = (f"parent confirmed by normal sibling(s) {', '.join(normal_other.smiles)}" if len(normal_other)
                      else f"deviation left after the parent shift ({shift:+.1f})")
            else:
                role = "ambiguous"
                ev = ("only radical of this parent in the set" if len(others) == 0
                      else "no normal sibling of another class (radical class or parent)")
            out.append({"row": int(r.row), "smiles": r.smiles, "role": role, "est_error_kcal": round(float(d), 1),
                        "Hf_label": float(r.Hf_radical),
                        "evidence": f"BDE {r.BDE:.1f} vs class median {r.BDE - r.dev_median:.1f} "
                                    f"[{r.audit_type[4:]}]; {ev}; parent {parent} (row {prow}, ΔHf {hp:.2f}); "
                                    f"siblings: {sib_txt}"})
    return out


# ───────────────────────── main ─────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--smiles_col", default="smiles")
    ap.add_argument("--h_col", default=None, help="ΔHf column (default: first column whose name contains 'hf')")
    ap.add_argument("--s_col", default=None, help="S298 column for the duplicate report (default: auto)")
    ap.add_argument("--h_unit", choices=["kcal", "kj"], default="kcal")
    ap.add_argument("--tol", type=float, default=12.0, help="flag if |BDE − typical| exceeds this (kcal/mol)")
    ap.add_argument("--zmax", type=float, default=4.0, help="flag if the robust z-score within its class exceeds this")
    ap.add_argument("--tol_sib", type=float, default=8.0,
                    help="a sibling within this of its class median counts as normal (kcal/mol)")
    ap.add_argument("--dup_tol", type=float, default=1.0, help="report duplicates whose ΔHf differ by more than this")
    ap.add_argument("--out_prefix", default="bde_audit")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    scol = next(c for c in df.columns if str(c).lower() == args.smiles_col.lower())
    hcol = args.h_col or next(c for c in df.columns if "hf" in str(c).lower())
    s_col = args.s_col or next((c for c in df.columns if str(c).lower().startswith("s_") or str(c).lower() == "s298"),
                               None)
    h = pd.to_numeric(df[hcol], errors="coerce").to_numpy(float) * (1.0 / 4.184 if args.h_unit == "kj" else 1.0)
    s = pd.to_numeric(df[s_col], errors="coerce").to_numpy(float) if s_col else np.full(len(df), np.nan)
    smiles = df[scol].astype(str).str.strip().tolist()
    mols = [Chem.MolFromSmiles(x) for x in smiles]
    print(f"{args.csv}: {len(df)} rows | ΔHf column '{hcol}' ({args.h_unit}) -> kcal/mol")

    by_canon: dict[str, tuple[int, float]] = {}
    for i, m in enumerate(mols):
        if m is not None and np.isfinite(h[i]):
            by_canon.setdefault(Chem.MolToSmiles(m), (i, h[i]))

    # ── 1. BDE audit ──
    rows, n_rad = [], 0
    for i, m in enumerate(mols):
        if m is None or not np.isfinite(h[i]) or sum(a.GetNumRadicalElectrons() for a in m.GetAtoms()) != 1:
            continue
        n_rad += 1
        parent, typ, atyp = parent_of(m)
        if parent is None or atyp is None or parent not in by_canon:
            continue
        j, h_parent = by_canon[parent]
        rows.append({"row": i, "smiles": smiles[i], "type": typ, "audit_type": atyp, "parent": parent,
                     "parent_row": j, "Hf_radical": h[i], "Hf_parent": h_parent,
                     "BDE": h[i] + HF_H_ATOM - h_parent})
    res = pd.DataFrame(rows)
    print(f"mono-radicals: {n_rad} | with parent in the dataset: {len(res)}\n")
    if res.empty:
        return

    print(f"  {'audit class':<17s} {'n':>5s} {'median':>8s} {'MAD':>6s} {'min':>8s} {'max':>8s} {'typical':>8s}")
    stats = {}
    for typ, g in res.groupby("audit_type"):
        med = float(g.BDE.median())
        mad = float(1.4826 * np.median(np.abs(g.BDE - med))) or 1.0
        stats[typ] = (med, mad)
        print(f"  {typ:<17s} {len(g):>5d} {med:>8.1f} {mad:>6.1f} {g.BDE.min():>8.1f} {g.BDE.max():>8.1f} "
              f"{TYPICAL.get(typ, float('nan')):>8.1f}")

    res["dev_typical"] = res.apply(lambda r: r.BDE - TYPICAL.get(r.audit_type, np.nan), axis=1)
    res["dev_median"] = res.apply(lambda r: r.BDE - stats[r.audit_type][0], axis=1)
    res["robust_z"] = res.apply(lambda r: r.dev_median / stats[r.audit_type][1], axis=1)
    flagged = res[(res.dev_typical.abs() > args.tol) | (res.robust_z.abs() > args.zmax)].copy()
    flagged = flagged.reindex(flagged.dev_median.abs().sort_values(ascending=False).index)
    print(f"\nflagged: {len(flagged)} BDEs (|BDE − typical| > {args.tol} or |robust z| > {args.zmax})")

    # ── 2. attribution ──
    sus = attribute(res, args.tol, args.tol_sib, args.zmax)

    # ── 3. resonance / stereo duplicates ──
    def _k(smi):
        return skeleton_key(Chem.MolFromSmiles(smi))
    key_ok = (_k("C=C[O]") == _k("[CH2]C=O") and _k("C=CC/C=C/C") == _k("C=CCC=CC")
              and _k("CCCO") != _k("CC#CO") and _k("C=C1[CH]CCC1") != _k("[CH2]C1=CC#CC1")
              and _k("C=CCC") != _k("[CH2]CC=C"))
    groups = defaultdict(list)
    if not key_ok:
        print("\nWARNING: skeleton_key self-check failed with this RDKit version — duplicate search skipped")
    for i, m in enumerate(mols if key_ok else []):
        if m is not None and (k := skeleton_key(m)) is not None:
            groups[k].append(i)
    dup_rows = []
    for k, idx in groups.items():
        if len(idx) < 2:
            continue
        hs = h[idx]
        spread = float(np.nanmax(hs) - np.nanmin(hs))
        dup_rows.append({"rows": " ".join(map(str, idx)), "smiles": " | ".join(smiles[i] for i in idx),
                         "Hf": " | ".join(f"{x:.2f}" for x in hs), "S298": " | ".join(f"{x:.2f}" for x in s[idx]),
                         "Hf_spread": round(spread, 2)})
        if spread > args.dup_tol:
            for i in idx:
                sus.append({"row": i, "smiles": smiles[i], "role": "duplicate", "est_error_kcal": np.nan,
                            "Hf_label": float(h[i]),
                            "evidence": f"same species as rows {', '.join(str(j) for j in idx if j != i)} "
                                        f"({' | '.join(smiles[j] for j in idx)}); ΔHf spread {spread:.1f}"})
    dups = pd.DataFrame(dup_rows)
    if not dups.empty:
        dups = dups.sort_values("Hf_spread", ascending=False)

    sus = pd.DataFrame(sus, columns=["row", "smiles", "role", "est_error_kcal", "Hf_label", "evidence"])
    if not sus.empty:
        order = {"parent": 0, "radical": 1, "duplicate": 2, "ambiguous": 3}
        sus["_o"] = sus.role.map(order)
        sus["_e"] = -sus.est_error_kcal.abs().fillna(0)
        sus = sus.sort_values(["_o", "_e"]).drop(columns=["_o", "_e"]).drop_duplicates(["row", "role"])

    print(f"\nsuspected labels: {len(sus)}  "
          + "  ".join(f"{r}: {int((sus.role == r).sum())}" for r in ("parent", "radical", "duplicate", "ambiguous")))
    for role in ("parent", "radical", "duplicate", "ambiguous"):
        part = sus[sus.role == role]
        if part.empty:
            continue
        print(f"\n  ── {role} ──")
        for _, r in part.head(40).iterrows():
            e = "" if not np.isfinite(r.est_error_kcal) else f"est. error {r.est_error_kcal:+6.1f}"
            print(f"  row {r.row:>5d}  {r.smiles:<30s} ΔHf {r.Hf_label:8.2f}  {e}\n           {r.evidence}")
    if not dups.empty:
        print(f"\nduplicate species (resonance / stereo): {len(dups)} groups, "
              f"{int((dups.Hf_spread > args.dup_tol).sum())} with ΔHf spread > {args.dup_tol} kcal/mol")
        for _, r in dups.head(20).iterrows():
            print(f"  rows {r.rows:<14s} {r.smiles:<40s} ΔHf {r.Hf:<22s} S {r.S298}")

    res.to_csv(f"{args.out_prefix}_all.csv", index=False)
    flagged.to_csv(f"{args.out_prefix}_flagged.csv", index=False)
    sus.to_csv(f"{args.out_prefix}_suspects.csv", index=False)
    dups.to_csv(f"{args.out_prefix}_duplicates.csv", index=False)
    print(f"\nwrote {args.out_prefix}_all.csv, _flagged.csv, _suspects.csv, _duplicates.csv")


if __name__ == "__main__":
    main()
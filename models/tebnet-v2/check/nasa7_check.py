"""C5: is TEB-Net's NASA-7 output exactly consistent, and does it stay physical up to T_high?

Runs trained checkpoints (no training) on their own test set and checks the 14 NASA-7 coefficients that would
go into a Chemkin / Cantera thermo file. For a decoupled model the ΔH model gives H298 and the S/Cp model gives
a1–a5 and S298; several run dirs per side = an ensemble (equal weights on the layer inputs, which is the same as
averaging the outputs, because the analytic layer is affine in its inputs).

What is checked
  1. Round trip: the polynomials reproduce the evaluated ΔH298, S298 and Cp(T_grid), in float64 and after
     writing the coefficients as Chemkin E15.8 text. (dH/dT = Cp and dS/dT = Cp/T inside one range hold for ANY
     NASA-7 polynomial, so they are printed only as a sanity check of the evaluation code, not as evidence.)
  2. Continuity at T_mid of Cp, H and S (enforced by the layer) and of dCp/dT (NOT enforced: reported).
  3. Physics beyond the data. The DeepTherm grid stops at 1500 K, so above T_mid the Cp prior has two points
     (1000, 1500 K) and is linear in T; up to T_high (3000 K) Cp is extrapolated. Checked against the classical
     (equipartition) limit of an ideal gas: Cp_inf = (3N − 2) R non-linear, (3N − 1.5) R linear, 2.5 R atom;
     a real molecule approaches it from below (free internal rotors lower it by R/2 each). Also: Cp must not
     decrease with T.
  4. Optional --complete_high: rebuilds only the high-T range so that it keeps Cp(T_mid), dCp/dT(T_mid) and the
     model's Cp at every grid temperature above T_mid (so every evaluated number is unchanged), and above the
     last grid temperature follows a monotone curve that saturates at the classical limit
     Cp_inf_est = (3N − 2 − n_rotors/2) R (the estimate RMG uses for Wilhoit extrapolation). H and S stay
     continuous. Checks 1–3 are repeated on the completed coefficients.

Usage (ensemble random0, decoupled ΔH / S-Cp models)
  B=benchmark_results_deeptherm
  python check/nasa7_check.py \
      --h_runs   $B/thermo_physics_chebyshev_3d_grouped_d5_global_w1-0-0_mon-h/random0/member*_seed* \
      --scp_runs $B/thermo_physics_chebyshev_3d_grouped_d5_global_w0-1.5-1_mon-scp/random0/member*_seed* \
      --complete_high --write_dat c5_random0.dat
  (one multitask model: --h_runs DIR only. If the CSV path stored in results.json is relative to another
   working directory, pass --csv; likewise --physics_cache.)
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
R = 8.314462618
J_TO_KCAL = 1.0 / 4184.0
def cp_r(a, T):
    T = np.asarray(T, dtype=np.float64)
    return a[:, :5] @ np.vstack([T ** k for k in range(5)])
def dcp_r(a, T):
    T = np.asarray(T, dtype=np.float64)
    return a[:, 1:5] @ np.vstack([k * T ** (k - 1) for k in range(1, 5)])
def h_rt(a, T):
    T = np.asarray(T, dtype=np.float64)
    return a[:, :5] @ np.vstack([T ** k / (k + 1) for k in range(5)]) + a[:, 5:6] / T
def s_r(a, T):
    T = np.asarray(T, dtype=np.float64)
    basis = [np.log(T), T, T ** 2 / 2, T ** 3 / 3, T ** 4 / 4]
    return a[:, :5] @ np.vstack(basis) + a[:, 6:7]
def piecewise(fn, lo, hi, T, t_mid):
    T = np.asarray(T, dtype=np.float64)
    return np.where(T[None, :] <= t_mid, fn(lo, T), fn(hi, T))
def props(lo, hi, T, t_mid):
    """Cp, H, S in J units on T (n, m)."""
    T = np.asarray(T, dtype=np.float64)
    return (R * piecewise(cp_r, lo, hi, T, t_mid),
            R * T[None, :] * piecewise(h_rt, lo, hi, T, t_mid),
            R * piecewise(s_r, lo, hi, T, t_mid))
def fill_a6_a7_high(lo, hi, t_mid):
    """a6_h, a7_h from H and S continuity at t_mid (the layer's step (c)); returns a new hi."""
    hi = hi.copy()
    tm = np.array([t_mid])
    h_mid = h_rt(lo, tm)[:, 0]
    s_mid = s_r(lo, tm)[:, 0]
    poly_h = hi[:, :5] @ np.array([t_mid ** k / (k + 1) for k in range(5)])
    poly_s = hi[:, :5] @ np.array([np.log(t_mid), t_mid, t_mid ** 2 / 2, t_mid ** 3 / 3, t_mid ** 4 / 4])
    hi[:, 5] = t_mid * (h_mid - poly_h)
    hi[:, 6] = s_mid - poly_s
    return hi
def layer_np(a15_low, a15_high, h298, s298, t_ref, t_mid):
    """numpy replica of NASA7ThermodynamicLayer (cross-check of the torch layer)."""
    n = len(h298)
    lo = np.zeros((n, 7))
    lo[:, :5] = a15_low
    lo[:, 5] = t_ref * (h298 / (R * t_ref) - a15_low @ np.array([t_ref ** k / (k + 1) for k in range(5)]))
    lo[:, 6] = s298 / R - a15_low @ np.array([np.log(t_ref), t_ref, t_ref ** 2 / 2, t_ref ** 3 / 3, t_ref ** 4 / 4])
    hi = np.zeros((n, 7))
    hi[:, 1:5] = a15_high[:, 1:5]
    cp_mid = cp_r(lo, [t_mid])[:, 0]
    hi[:, 0] = cp_mid - a15_high[:, 1:5] @ np.array([t_mid ** k for k in range(1, 5)])
    return lo, fill_a6_a7_high(lo, hi, t_mid)
def classical_limits(n_atoms, is_linear, n_rotors):
    """(hard bound, RMG-style estimate) of Cp at T -> inf, J/(mol K)."""
    n_atoms = np.asarray(n_atoms, dtype=np.float64)
    lin = np.asarray(is_linear) >= 0.5
    mono = n_atoms <= 1
    hard = np.where(mono, 2.5, np.where(lin, 3 * n_atoms - 1.5, 3 * n_atoms - 2.0)) * R
    est = hard - np.where(mono | lin, 0.0, 0.5 * np.asarray(n_rotors, dtype=np.float64)) * R
    return hard, np.maximum(est, np.where(mono, 2.5, np.where(lin, 3.5, 4.0)) * R)
def diagnostics(lo, hi, t_mid, t_high, hard, est, t_min=300.0, step=10.0):
    T = np.unique(np.concatenate([np.arange(t_min, t_high + 0.5 * step, step), [t_mid]]))
    cp = R * piecewise(cp_r, lo, hi, T, t_mid)
    dcp = R * piecewise(dcp_r, lo, hi, T, t_mid)
    over = cp > hard[:, None]
    first = np.where(over.any(1), T[np.argmax(over, axis=1)], np.nan)
    return {
        "n": len(hard),
        "over_hard": int(over.any(1).sum()),
        "over_hard_1R": int((cp > hard[:, None] + R).any(1).sum()),
        "first_T_median": float(np.nanmedian(first)) if np.isfinite(first).any() else float("nan"),
        "decreasing": int((dcp < -1e-6).any(1).sum()),
        "ratio_est_at_high": np.percentile(cp[:, -1] / est, [5, 50, 95]),
        "cp_high_minus_hard_max": float((cp[:, -1] - hard).max()),
    }
def chemkin_round(a):
    return np.vectorize(lambda x: float(f"{x:15.8E}"))(a)
def consistency(lo, hi, t_mid, t_ref, t_grid, h_eval, s_eval, cp_eval):
    out = {}
    for name, (l, h) in {"float64": (lo, hi), "Chemkin E15.8": (chemkin_round(lo), chemkin_round(hi))}.items():
        cp, H, S = props(l, h, t_grid, t_mid)
        h298 = R * t_ref * h_rt(l, [t_ref])[:, 0]
        s298 = R * s_r(l, [t_ref])[:, 0]
        out[name] = (np.abs(h298 - h_eval).max(), np.abs(s298 - s_eval).max(), np.abs(cp - cp_eval).max())
    tm = np.array([t_mid])
    jumps = {
        "Cp [J/(mol K)]": np.abs(R * (cp_r(lo, tm) - cp_r(hi, tm))).max(),
        "H [J/mol]": np.abs(R * t_mid * (h_rt(lo, tm) - h_rt(hi, tm))).max(),
        "S [J/(mol K)]": np.abs(R * (s_r(lo, tm) - s_r(hi, tm))).max(),
    }
    d_lo, d_hi = R * dcp_r(lo, tm)[:, 0], R * dcp_r(hi, tm)[:, 0]
    kink = np.abs(d_hi - d_lo)
    rel_kink = kink / np.maximum(np.abs(d_lo), 1e-12)
    ident = 0.0
    for a, T in ((lo, np.linspace(400, t_mid - 100, 9)), (hi, np.linspace(t_mid + 100, 2900, 9))):
        e = 1e-3
        dH = R * ((T + e) * h_rt(a, T + e) - (T - e) * h_rt(a, T - e)) / (2 * e)
        dS = R * (s_r(a, T + e) - s_r(a, T - e)) / (2 * e)
        cp = R * cp_r(a, T)
        ident = max(ident, np.abs(dH / cp - 1).max(), np.abs(dS * T / cp - 1).max())
    return out, jumps, (np.median(kink), kink.max(), np.median(rel_kink)), ident
def complete_high(lo, hi, t_mid, t_data, hard, est, t_high, n_fit=30, scale=1000.0):
    """New high-range coefficients a1..a5 (a6, a7 then from H and S continuity).

    Target above T1 = max(t_data): a monotone curve that starts at the model's Cp and slope at T1 (C1, s1, from
    the trained high-range polynomial) and saturates at the classical limit,
        Cp_ext(T) = Cp_inf − (Cp_inf − C1)(T1/T)^p,   p = s1 T1 / (Cp_inf − C1),
    with Cp_inf the RMG-style estimate (the hard bound if C1 is already above the estimate).
    Equalities: Cp and dCp/dT continuous at t_mid; Cp at every t in t_data unchanged (so every evaluated number
    stays the same); Cp and dCp/dT at t_high equal to Cp_ext. With one grid point above t_mid (DeepTherm) these
    are exactly 5 conditions for 5 coefficients; with more points the t_high conditions move into a least-squares
    fit to Cp_ext on (T1, t_high]. (Tested on 300 rigid-rotor/harmonic-oscillator molecules: no crossing of the
    classical limit, no decreasing Cp, Cp(2000–3000 K) within ~2–4 J/(mol K) on average of the exact curve,
    against 15–60 for the linear extrapolation.)
    Returns (hi_new, n_flat): n_flat = molecules with no saturating target (C1 above the hard bound or s1 <= 0);
    they get a flat target at C1."""
    t_data = np.atleast_1d(np.asarray(t_data, dtype=np.float64))
    T1 = float(t_data.max())
    T_fit = np.linspace(T1, t_high, n_fit + 1)[1:]
    c1 = R * cp_r(hi, [T1])[:, 0]
    s1 = R * dcp_r(hi, [T1])[:, 0]
    cp_mid = cp_r(lo, [t_mid])[:, 0]
    dcp_mid = dcp_r(lo, [t_mid])[:, 0]
    cp_dat = cp_r(hi, t_data)
    k = np.arange(5)
    def vrow(T):
        return (np.asarray(T, dtype=np.float64)[:, None] / scale) ** k
    def drow(T):
        return np.where(k > 0, k * (T / scale) ** np.maximum(k - 1, 0), 0.0) / scale
    end_as_equality = len(t_data) + 4 <= 5
    hi_new = hi.copy()
    n_flat = 0
    for i in range(len(hard)):
        cinf = est[i] if c1[i] < est[i] else hard[i]
        if c1[i] >= cinf or s1[i] <= 0:
            f_end, s_end = c1[i], 0.0
            y = np.full(len(T_fit), c1[i])
            n_flat += 1
        else:
            p = s1[i] * T1 / (cinf - c1[i])
            y = cinf - (cinf - c1[i]) * (T1 / T_fit) ** p
            f_end = y[-1]
            s_end = (cinf - c1[i]) * p * T1 ** p * t_high ** (-p - 1)
        C = [vrow([t_mid])[0], drow(t_mid), *vrow(t_data)]
        d = [cp_mid[i], dcp_mid[i], *cp_dat[i]]
        A, yy = vrow(T_fit), y / R
        if end_as_equality:
            C += [vrow([t_high])[0], drow(t_high)]
            d += [f_end / R, s_end / R]
        else:
            A = np.vstack([A, 10.0 * drow(t_high)])
            yy = np.concatenate([yy, [10.0 * s_end / R]])
        C, d = np.array(C), np.array(d)
        n_c = len(d)
        K = np.block([[2 * A.T @ A, C.T], [C, np.zeros((n_c, n_c))]])
        rhs = np.concatenate([2 * A.T @ yy, d])
        b = np.linalg.lstsq(K, rhs, rcond=None)[0][:5]
        hi_new[i, :5] = b / scale ** k
    return fill_a6_a7_high(lo, hi_new, t_mid), n_flat
def write_dat(path, names, comps, lo, hi, t_low, t_mid, t_high):
    lines = ["THERMO", f"{t_low:10.3f}{t_mid:10.3f}{t_high:10.3f}"]
    for name, comp, l, h in zip(names, comps, lo, hi):
        el = "".join(f"{e:<2s}{c:>3d}" for e, c in list(comp.items())[:4])
        lines.append(f"{name:<24s}{el:<20s}G{t_low:10.3f}{t_high:10.3f}{t_mid:8.3f}{'':6s}1")
        lines.append("".join(f"{x:15.8E}" for x in h[:5]) + "    2")
        lines.append("".join(f"{x:15.8E}" for x in (h[5], h[6], l[0], l[1], l[2])) + "    3")
        lines.append("".join(f"{x:15.8E}" for x in l[3:7]) + " " * 15 + "    4")
    lines.append("END")
    Path(path).write_text("\n".join(lines) + "\n")
def run_models(run_dirs, csv_override, cache_override, batch_size):
    """Layer inputs (float64, captured with a forward hook) and evaluated outputs (as scored) per run."""
    here = Path(__file__).resolve().parent
    for p in (here, here.parent / "scripts"):
        if (p / "train.py").exists():
            sys.path.insert(0, str(p))
            break
    import torch
    from torch.utils.data import DataLoader
    from train import make_atom_featurizer, morgan_bits, COL
    from tebnet.data.dataset import ThermochemDataset
    from tebnet.data.collate import ThermoCollater
    from tebnet.models.tebnet import TEBNetV20
    loaders, runs = {}, []
    for rd in run_dirs:
        rd = Path(rd)
        res = json.loads((rd / "results.json").read_text())
        a = res["args"]
        t_grid = np.asarray(res["t_grid"], dtype=np.float64)
        test_idx = tuple(res["test_indices"])
        csv = csv_override or a["csv"]
        cache = cache_override or a.get("physics_cache", "none")
        cache = None if str(cache).lower() == "none" else cache
        use_3d = not a.get("no_3d", False)
        key = (csv, str(cache), use_3d, a["atom_featurizer"], a.get("ecfp_bits", 0), test_idx)
        if key not in loaders:
            full = ThermochemDataset.from_csv(
                csv, T_grid=t_grid, smiles_col=a["smiles_col"], physics_cache=cache,
                h_col=a.get("h_col"), s_col=a.get("s_col"),
                cp_cols=[c.strip() for c in a["cp_cols"].split(",")] if a.get("cp_cols") else None,
                h_unit=a.get("h_unit"), s_unit=a.get("s_unit"), cp_unit=a.get("cp_unit"),
                strict=True, compute_aux=False, compute_comp=False)
            ds = ThermochemDataset([full.records[i] for i in test_idx], T_grid=t_grid, aux_feature_set="v15",
                                   compute_aux=True, use_3d=use_3d)
            if a.get("ecfp_bits", 0) > 0:
                for dp in ds.datapoints:
                    dp.x_physic = np.concatenate([dp.x_physic, morgan_bits(dp.mol, a["ecfp_bits"])]).astype(np.float32)
            loaders[key] = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                      collate_fn=ThermoCollater(atom_featurizer=make_atom_featurizer(a["atom_featurizer"])))
        ckpt = Path(res.get("checkpoint") or "")
        if not ckpt.exists():
            ckpt = rd / "best.ckpt"
        model = TEBNetV20.load_from_checkpoint(str(ckpt), map_location="cpu").eval()
        cap = []
        hook = model.thermo_layer.register_forward_hook(
            lambda m, inp, out: cap.append([t.detach().double().cpu().numpy() for t in inp[:4]]))
        ev = {k: [] for k in ("h", "s", "cp", "n_atoms", "lin", "rot")}
        smiles = []
        with torch.no_grad():
            for batch in loaders[key]:
                o = model(batch)
                ev["h"].append(o["dh_f_298"].double().numpy())
                ev["s"].append(o["s_298"].double().numpy())
                ev["cp"].append(o["cp"].double().numpy())
                xp = batch.x_physic.double().numpy()
                ev["n_atoms"].append(xp[:, COL["n_atoms_H"]])
                ev["lin"].append(xp[:, COL["is_linear"]])
                ev["rot"].append(xp[:, COL["n_rotors"]])
                smiles.extend(batch.smiles)
        hook.remove()
        inp = [np.concatenate([c[j] for c in cap]) for j in range(4)]
        runs.append({"dir": str(rd), "t_grid": t_grid, "t_mid": float(a.get("t_mid", 1000.0)),
                     "t_high": float(a.get("t_high_dom", 3000.0)), "test_idx": test_idx, "smiles": smiles,
                     "a15_low": inp[0], "a15_high": inp[1], "h298": inp[2], "s298": inp[3],
                     **{k: np.concatenate(v) for k, v in ev.items()}})
        print(f"  loaded {rd} ({len(smiles)} test molecules)")
    return runs
def mean_of(runs, key):
    return np.mean([r[key] for r in runs], axis=0)
def report(title, lo, hi, meta, h_eval, s_eval, cp_eval, hard, est):
    t_mid, t_ref, t_grid, t_high = meta["t_mid"], 298.15, meta["t_grid"], meta["t_high"]
    rt, jumps, kink, ident = consistency(lo, hi, t_mid, t_ref, t_grid, h_eval, s_eval, cp_eval)
    dg = diagnostics(lo, hi, t_mid, t_high, hard, est)
    print(f"\n### {title} (n = {dg['n']})\n")
    print("| Check | ΔH298 [J/mol] | S298 [J/(mol K)] | Cp(T_grid) [J/(mol K)] |")
    print("|---|---|---|---|")
    for name, (eh, es, ec) in rt.items():
        print(f"| NASA-7 ({name}) vs evaluated prediction, max abs | {eh:.2e} | {es:.2e} | {ec:.2e} |")
    print(f"\nJumps at T_mid = {t_mid:g} K (max abs): " + ", ".join(f"{k} {v:.1e}" for k, v in jumps.items()))
    print(f"Kink in dCp/dT at T_mid (not enforced by the layer): median {kink[0]:.2e}, max {kink[1]:.2e} "
          f"J/(mol K²); median {100 * kink[2]:.0f} % of the low-side slope")
    print(f"Sanity, inside each range: max |dH/dT / Cp − 1|, |T dS/dT / Cp − 1| = {ident:.1e} "
          f"(holds for any NASA-7 polynomial)")
    n = dg["n"]
    r5, r50, r95 = dg["ratio_est_at_high"]
    print(f"\n| Physics, 300–{t_high:g} K | Molecules |")
    print("|---|---|")
    print(f"| Cp above the classical limit (3N−2)R somewhere | {dg['over_hard']} ({100 * dg['over_hard'] / n:.1f} %) |")
    print(f"| … by more than 1 R | {dg['over_hard_1R']} ({100 * dg['over_hard_1R'] / n:.1f} %) |")
    print(f"| median T of the first crossing | {dg['first_T_median']:.0f} K |")
    print(f"| Cp decreasing with T somewhere | {dg['decreasing']} ({100 * dg['decreasing'] / n:.1f} %) |")
    print(f"| Cp({t_high:g} K) / Cp_inf estimate, 5th / 50th / 95th pct | {r5:.2f} / {r50:.2f} / {r95:.2f} |")
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h_runs", nargs="+", required=True, help="run dir(s) of the ΔH (or multitask) model")
    ap.add_argument("--scp_runs", nargs="+", default=None, help="run dir(s) of the S/Cp model (decoupled)")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--physics_cache", default=None)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--complete_high", action="store_true")
    ap.add_argument("--write_dat", default=None, help="write the (completed, if asked) NASA-7 blocks here")
    args = ap.parse_args()
    print("H side:")
    h_runs = run_models(args.h_runs, args.csv, args.physics_cache, args.batch_size)
    print("S/Cp side:")
    scp_runs = run_models(args.scp_runs, args.csv, args.physics_cache, args.batch_size) if args.scp_runs else h_runs
    import torch
    from tebnet.nn.nasa_thermo import NASA7ThermodynamicLayer
    ref = h_runs[0]
    for r in h_runs + scp_runs:
        if r["test_idx"] != ref["test_idx"] or r["smiles"] != ref["smiles"]:
            raise SystemExit(f"{r['dir']} has a different test set from {ref['dir']}")
        if r["t_mid"] != ref["t_mid"] or not np.array_equal(r["t_grid"], ref["t_grid"]):
            raise SystemExit(f"{r['dir']} uses another T grid / T_mid")
    a15_low, a15_high = mean_of(scp_runs, "a15_low"), mean_of(scp_runs, "a15_high")
    h298, s298 = mean_of(h_runs, "h298"), mean_of(scp_runs, "s298")
    layer = NASA7ThermodynamicLayer(t_ref=298.15, t_mid=ref["t_mid"])
    out = layer(*(torch.from_numpy(x) for x in (a15_low, a15_high, h298, s298)),
                torch.from_numpy(ref["t_grid"]))
    lo, hi = out["coeffs_low"].numpy(), out["coeffs_high"].numpy()
    lo_np, hi_np = layer_np(a15_low, a15_high, h298, s298, 298.15, ref["t_mid"])
    print(f"\ntorch layer vs numpy replica: max |Δa| low {np.abs(lo - lo_np).max():.1e}, "
          f"high {np.abs(hi - hi_np).max():.1e}")
    h_eval, s_eval, cp_eval = mean_of(h_runs, "h"), mean_of(scp_runs, "s"), mean_of(scp_runs, "cp")
    hard, est = classical_limits(ref["n_atoms"], ref["lin"], ref["rot"])
    meta = {"t_mid": ref["t_mid"], "t_grid": ref["t_grid"], "t_high": ref["t_high"]}
    print(f"\n{len(h_runs)} ΔH model(s), {len(scp_runs)} S/Cp model(s); T grid {ref['t_grid'].tolist()} K, "
          f"T_mid {ref['t_mid']:g} K, NASA high range to {ref['t_high']:g} K")
    report("TEB-Net NASA-7 as trained", lo, hi, meta, h_eval, s_eval, cp_eval, hard, est)
    if args.complete_high:
        t_data = ref["t_grid"][ref["t_grid"] > ref["t_mid"]]
        hi_c, n_flat = complete_high(lo, hi, ref["t_mid"], t_data, hard, est, ref["t_high"])
        report("after --complete_high", lo, hi_c, meta, h_eval, s_eval, cp_eval, hard, est)
        T_between = np.linspace(ref["t_mid"], t_data.max(), 21)
        d = np.abs(R * (cp_r(hi_c, T_between) - cp_r(hi, T_between))).max(1)
        _, h0, s0 = props(lo, hi, [ref["t_high"]], ref["t_mid"])
        _, h1, s1 = props(lo, hi_c, [ref["t_high"]], ref["t_mid"])
        dh, ds = np.abs(h1 - h0)[:, 0] * J_TO_KCAL, np.abs(s1 - s0)[:, 0] / 4.184
        print(f"\nno saturating target (flat at Cp(T1)): {n_flat} molecules | change of Cp between "
              f"{ref['t_mid']:g} and {t_data.max():g} K (no data there): median {np.median(d):.2f}, "
              f"max {d.max():.2f} J/(mol K)")
        print(f"change at {ref['t_high']:g} K, completed − as trained: |ΔH| median {np.median(dh):.2f}, "
              f"max {dh.max():.2f} kcal/mol; |ΔS| median {np.median(ds):.2f}, max {ds.max():.2f} cal/(mol K)")
        hi_trained, hi = hi, hi_c
    else:
        hi_trained = None
    if args.write_dat:
        from rdkit import Chem
        comps = []
        for s in ref["smiles"]:
            m = Chem.AddHs(Chem.MolFromSmiles(s))
            c = {}
            for at in m.GetAtoms():
                c[at.GetSymbol().upper()] = c.get(at.GetSymbol().upper(), 0) + 1
            comps.append(c)
        names = [f"TEB{i}" for i in ref["test_idx"]]
        t_low = float(ref["t_grid"].min())
        write_dat(args.write_dat, names, comps, lo, hi, t_low, ref["t_mid"], ref["t_high"])
        Path(args.write_dat).with_suffix(".smiles.csv").write_text(
            "name,smiles\n" + "\n".join(f"{n},{s}" for n, s in zip(names, ref["smiles"])) + "\n")
        print(f"\nwrote {len(names)} NASA-7 blocks to {args.write_dat} (+ name→SMILES table)")
        if hi_trained is not None:
            p = Path(args.write_dat).with_suffix(".as_trained.dat")
            write_dat(p, names, comps, lo, hi_trained, t_low, ref["t_mid"], ref["t_high"])
            print(f"wrote the as-trained (linear high range) blocks to {p}")
if __name__ == "__main__":
    main()

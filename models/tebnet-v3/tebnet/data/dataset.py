from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from rdkit import Chem
from tebnet.data.featurizer import (
    ThermoAtomFeaturizer,
    compute_auxiliary_features,
    compute_composition_vector,
    compute_v15_vector,
    physics_cache_path,
    V15_NAMES,
    COMP_DIM,
    FEATURES_VERSION,
)
from tebnet.data.parsers.utils import get_default_temperature_grid
import pandas as pd
import re as _re

_H_FACTOR = {"kcal": 4184.0, "kj": 1000.0, "j": 1.0}
_SCP_FACTOR = {"cal": 4.184, "j": 1.0, "kcal": 4184.0, "kj": 1000.0}
_SKIP = ("unc", "err", "std", "sigma", "source", "ref", "method", "note", "flag", "weight")
logger = logging.getLogger(__name__)

def _finite_or_nan(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if np.isfinite(v) else float("nan")

@dataclass
class ThermoDatapoint:
    smiles: str
    """SMILES string."""
    mol: Chem.Mol | None
    """RDKit molecule."""
    cp_grid: np.ndarray
    """Cp values at temperature grid, shape (n_temps,), in J/(mol·K). NaN = missing."""
    dh_f_298: float
    """ΔH_f(298.15) in J/mol. NaN = missing."""
    s_298: float
    """S(298.15) in J/(mol·K). NaN = missing."""
    T_grid: np.ndarray
    """Temperature grid, shape (n_temps,)."""
    weight: float = 1.0
    """Sample weight (based on data quality)."""
    name: str = ""
    """Species name/identifier."""
    x_physic: np.ndarray | None = None
    """Auxiliary descriptor vector (v14: 4/8/16-D; v15: len(V15_NAMES) named vector)."""
    comp: np.ndarray | None = None
    """Element-count vector (COMP_DIM,)."""

class ThermochemDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        T_grid: np.ndarray | None = None,
        compute_aux: bool = True,
        aux_feature_set: str = "extended",   # "minimal" | "extended" | "routed" | "v15"
        compute_comp: bool = True,
        use_3d: bool = True,                 # v15 only: compute MOI features if not cached
        atom_featurizer: ThermoAtomFeaturizer | None = None,  # kept for backward compat; unused
    ):
        self.T_grid = T_grid if T_grid is not None else get_default_temperature_grid()
        self.compute_aux = compute_aux
        self.aux_feature_set = aux_feature_set
        self.compute_comp = compute_comp
        self.use_3d = use_3d
        self.datapoints: list[ThermoDatapoint] = []
        self.records = records
        self.n_from_cache = 0
        self._process_records(records)

    def _process_records(self, records: list[dict]) -> None:
        n_skipped = 0
        for rec in records:
            smiles = rec.get("smiles")
            if smiles is None:
                n_skipped += 1
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                n_skipped += 1
                logger.debug(f"Invalid SMILES skipped: {smiles}")
                continue

            cp_raw = np.asarray(
                rec.get("Cp", np.full(len(self.T_grid), np.nan)),
                dtype=np.float32,
            )
            cp = np.full(len(self.T_grid), np.nan, dtype=np.float32)
            min_len = min(len(cp_raw), len(self.T_grid))
            cp[:min_len] = cp_raw[:min_len]

            dh_f_298 = _finite_or_nan(rec.get("dH_f_298", np.nan))
            s_298 = _finite_or_nan(rec.get("S_298", np.nan))
            w = _finite_or_nan(rec.get("weight", 1.0))
            weight = 1.0 if np.isnan(w) else w
            name = rec.get("name", smiles)

            x_physic, comp = None, None
            if self.compute_aux and self.aux_feature_set == "v15":
                vec = rec.get("x_v15")
                if vec is not None and len(vec) == len(V15_NAMES):
                    x_physic = np.asarray(vec, dtype=np.float32)
                    self.n_from_cache += 1
                else:
                    x_physic = compute_v15_vector(mol, use_3d=self.use_3d)
                comp = x_physic[:COMP_DIM].copy()
            else:
                if self.compute_aux:
                    try:
                        x_physic = compute_auxiliary_features(
                            mol, feature_set=self.aux_feature_set
                        )
                    except Exception as e:
                        logger.debug(f"Skipping SMILES {smiles} due to physics error: {e}")
                        n_skipped += 1
                        continue
                if self.compute_comp:
                    try:
                        comp = compute_composition_vector(mol)
                    except Exception as e:
                        logger.debug(f"Composition failed for {smiles}: {e}")
                        comp = None

            self.datapoints.append(ThermoDatapoint(
                smiles=smiles,
                mol=mol,
                cp_grid=cp,
                dh_f_298=dh_f_298,
                s_298=s_298,
                T_grid=self.T_grid,
                weight=weight,
                name=name,
                x_physic=x_physic,
                comp=comp,
            ))

        if n_skipped > 0:
            logger.warning(f"Skipped {n_skipped} invalid records")
        msg = f"Dataset created with {len(self.datapoints)} molecules"
        if self.aux_feature_set == "v15" and self.compute_aux:
            msg += f" (v15 features: {self.n_from_cache} from cache, " \
                   f"{len(self.datapoints) - self.n_from_cache} computed on the fly)"
        logger.info(msg)

    def __len__(self) -> int:
        return len(self.datapoints)

    def __getitem__(self, idx: int) -> dict:
        dp = self.datapoints[idx]
        result = {
            "smiles": dp.smiles,
            "mol": dp.mol,
            "cp_grid": torch.tensor(dp.cp_grid, dtype=torch.float32),
            "dh_f_298": torch.tensor(dp.dh_f_298, dtype=torch.float32),
            "s_298": torch.tensor(dp.s_298, dtype=torch.float32),
            "weight": dp.weight,
            "T_grid": torch.tensor(dp.T_grid, dtype=torch.float32),
        }
        result["x_physic"] = (
            torch.tensor(dp.x_physic, dtype=torch.float32) if dp.x_physic is not None else None
        )
        result["comp"] = torch.tensor(dp.comp, dtype=torch.float32) if dp.comp is not None else None
        return result

    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_physics_cache(cache: str | Path | None, csv_path: Path) -> dict[str, np.ndarray]:
        if cache is None:
            return {}
        path = physics_cache_path(csv_path) if str(cache) == "auto" else Path(cache)
        if not path.exists():
            if str(cache) != "auto":
                logger.warning(f"physics cache not found: {path} — features will be computed on the fly")
            else:
                logger.info(f"no physics cache at {path} — features will be computed on the fly "
                            f"(run data/compute_physics_features.py once to speed this up)")
            return {}
        dfc = pd.read_csv(path)
        if "features_version" in dfc.columns and (dfc["features_version"].astype(str) != FEATURES_VERSION).any():
            logger.warning(f"physics cache {path} was built with a different FEATURES_VERSION "
                           f"(expected {FEATURES_VERSION}) — ignoring it; rebuild the cache")
            return {}
        missing = [n for n in V15_NAMES if n not in dfc.columns]
        if missing:
            logger.warning(f"physics cache {path} lacks columns {missing} — ignoring it; rebuild the cache")
            return {}
        arr = dfc[V15_NAMES].to_numpy(dtype=np.float32)
        keys = dfc["smiles"].astype(str).str.strip().tolist()
        logger.info(f"loaded physics cache {path} ({len(keys)} species)")
        return {k: arr[i] for i, k in enumerate(keys)}

    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path,
        T_grid: np.ndarray | None = None,
        smiles_col: str = "smiles",
        physics_cache: str | Path | None = "auto",
        h_col: str | None = None,
        s_col: str | None = None,
        cp_cols: list[str] | None = None,
        h_unit: str | None = None,
        s_unit: str | None = None,
        cp_unit: str | None = None,
        strict: bool = False,
        **kwargs,
    ) -> "ThermochemDataset":
        """Load a thermo CSV. Column names and units are auto-detected (see _resolve_columns);
        override with h_col / s_col / cp_cols (aligned with T_grid) and
        h_unit in {kcal,kj,j}, s_unit / cp_unit in {cal,j}.
        strict=True raises (listing the CSV columns) if ΔH_f, S or Cp is entirely missing.
        """
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path)
        if T_grid is None:
            T_grid = get_default_temperature_grid()
        T_grid = np.asarray(T_grid, dtype=np.float64)
        if smiles_col not in df.columns:
            alt = [c for c in df.columns if str(c).strip().lower() == smiles_col.lower()]
            if not alt:
                raise ValueError(f"{csv_path.name}: no SMILES column '{smiles_col}'. Columns: {list(df.columns)}")
            smiles_col = alt[0]
        cache = cls._load_physics_cache(physics_cache, csv_path)

        cols = _resolve_columns(df, T_grid, h_col, s_col, cp_cols)
        n = len(df)

        # ── ΔH_f,298 → J/mol ──
        if cols["h"] is not None:
            raw = pd.to_numeric(df[cols["h"]], errors="coerce").to_numpy(np.float64)
            u = h_unit or _unit_from_name(cols["h"], "H") or _guess_unit(raw, "H", cols["h"])
            H = raw * _H_FACTOR[u]
        else:
            u, H = None, np.full(n, np.nan)
        units = {"H": u}

        # ── S298 → J/(mol·K) ──
        if cols["s"] is not None:
            raw = pd.to_numeric(df[cols["s"]], errors="coerce").to_numpy(np.float64)
            u = s_unit or _unit_from_name(cols["s"], "S") or _guess_unit(raw, "S", cols["s"])
            S = raw * _SCP_FACTOR[u]
        else:
            u, S = None, np.full(n, np.nan)
        units["S"] = u

        # ── Cp(T) → J/(mol·K) ──
        CP = np.full((n, len(T_grid)), np.nan)
        cp_found = [c for c in cols["cp"] if c is not None]
        if cp_found:
            raw_cols = []
            for j, c in enumerate(cols["cp"]):
                if c is not None:
                    CP[:, j] = pd.to_numeric(df[c], errors="coerce").to_numpy(np.float64)
                    raw_cols.append(c)
            u = cp_unit or _unit_from_name(raw_cols[0], "Cp") or _guess_unit(CP[:, 0], "Cp", raw_cols[0])
            CP *= _SCP_FACTOR[u]
            units["Cp"] = u
        else:
            units["Cp"] = None
            nasa_low = [f"a{i}_low" for i in range(1, 8)]
            nasa_high = [f"a{i}_high" for i in range(1, 8)]
            if all(c in df.columns for c in nasa_low + nasa_high):
                cl = df[nasa_low].to_numpy(np.float64)
                ch = df[nasa_high].to_numpy(np.float64)
                tmid = (pd.to_numeric(df["tmid_K"], errors="coerce").fillna(1000.0).to_numpy()
                        if "tmid_K" in df.columns else np.full(n, 1000.0))
                for j, T in enumerate(T_grid):
                    c = np.where((T <= tmid)[:, None], cl, ch)
                    CP[:, j] = (c[:, 0] + c[:, 1] * T + c[:, 2] * T**2 + c[:, 3] * T**3
                                + c[:, 4] * T**4) * 8.314462618
                units["Cp"] = "NASA-7 coefficients"
                cols["cp"] = ["<NASA-7>"] * len(T_grid)

        # ── report what was used ──
        def _med(a):
            a = a[np.isfinite(a)]
            return float(np.median(a)) if a.size else float("nan")
        logger.info(
            f"{csv_path.name}: column mapping\n"
            f"    ΔH_f,298 : {cols['h']!r} [{units['H']}]  -> median {_med(H)/1000:.1f} kJ/mol, "
            f"{np.isfinite(H).sum()}/{n} present\n"
            f"    S298     : {cols['s']!r} [{units['S']}]  -> median {_med(S):.1f} J/(mol·K), "
            f"{np.isfinite(S).sum()}/{n} present\n"
            f"    Cp(T)    : {cols['cp']} [{units['Cp']}]  -> median Cp({T_grid[0]:.0f}K) {_med(CP[:, 0]):.1f} J/(mol·K), "
            f"{np.isfinite(CP).all(1).sum()}/{n} complete rows"
        )
        problems = []
        if not np.isfinite(H).any():
            problems.append("ΔH_f,298")
        if not np.isfinite(S).any():
            problems.append("S298")
        if not np.isfinite(CP).any():
            problems.append("Cp")
        if problems:
            msg = (f"{csv_path.name}: could not find {', '.join(problems)}. CSV columns are:\n  {list(df.columns)}\n"
                   f"Pass them explicitly, e.g. --h_col <name> --s_col <name> "
                   f"--cp_cols <c300>,<c400>,... (one per T in the grid) and the units "
                   f"--h_unit kcal|kj|j --s_unit cal|j --cp_unit cal|j")
            if strict:
                raise ValueError(msg)
            logger.warning(msg)

        wcol = df["weight"] if "weight" in df.columns else None
        ncol = df["name"] if "name" in df.columns else None
        records = []
        for i in range(n):
            smi = df[smiles_col].iat[i]
            if smi is None or (isinstance(smi, float) and np.isnan(smi)):
                continue
            smi = str(smi).strip()
            w = wcol.iat[i] if wcol is not None else 1.0
            record = {
                "smiles": smi,
                "Cp": CP[i].astype(np.float32),
                "dH_f_298": float(H[i]),
                "S_298": float(S[i]),
                "weight": float(w) if pd.notna(w) else 1.0,
                "name": str(ncol.iat[i]) if ncol is not None and pd.notna(ncol.iat[i]) else smi,
            }
            if cache:
                record["x_v15"] = cache.get(smi)
            records.append(record)
        return cls(records, T_grid=T_grid, **kwargs)

def _norm(c) -> str:
    return _re.sub(r"[^a-z0-9]", "", str(c).lower())

def _unit_from_name(col: str, kind: str) -> str | None:
    n = str(col).lower().replace(" ", "")
    # "298kcal" / "300kcal" in an S or Cp name is "<T>K" + "cal", not kcal
    if ("kcal" in n) if kind == "H" else bool(_re.search(r"(?<![0-9])kcal", n)):
        return "kcal"
    if ("kj" in n) if kind == "H" else bool(_re.search(r"(?<![0-9])kj", n)):
        return "kj"
    if kind != "H" and "cal" in n:
        return "cal"
    if _re.search(r"(^|[^a-z])j($|[^a-z])|j/mol|jmol|j_mol", n):
        return "j"
    return None

def _guess_unit(values: np.ndarray, kind: str, col: str) -> str:
    v = np.abs(values[np.isfinite(values)])
    med = float(np.median(v)) if v.size else 0.0
    if kind == "H":
        u = "j" if med > 3000 else "kcal"
        hint = "--h_unit kcal|kj|j"
    elif kind == "S":
        u = "j" if med > 250 else "cal"
        hint = "--s_unit cal|j"
    else:
        u = "j" if med > 70 else "cal"
        hint = "--cp_unit cal|j"
    logger.warning(f"unit of column {col!r} not in its name; guessed '{u}' from magnitudes "
                   f"(median |value| = {med:.3g}). If wrong, pass {hint}.")
    return u

def _resolve_columns(df: pd.DataFrame, T_grid: np.ndarray, h_col=None, s_col=None, cp_cols=None) -> dict:
    cols = list(df.columns)

    def _check(c):
        if c is not None and c not in df.columns:
            raise ValueError(f"column {c!r} not in CSV. Columns: {cols}")
        return c

    # ΔH_f,298
    if h_col is None:
        legacy = [c for c in ("Hf298_kJ_mol", "Hf_298.15K_kcal/mol", "dH_f_298") if c in df.columns]
        cand = legacy or [
            c for c in cols
            if not any(k in _norm(c) for k in _SKIP)
            and (_norm(c).startswith(("hf", "dhf", "deltahf", "dfh", "h298", "dh298", "enthalpy"))
                 or "enthalpy" in _norm(c) or "hf298" in _norm(c))
        ]
        if len(cand) > 1:
            logger.warning(f"several ΔH_f candidates {cand}; using {cand[0]!r} (override with --h_col)")
        h_col = cand[0] if cand else None
    # S298
    if s_col is None:
        legacy = [c for c in ("S_298_J_mol_K", "S_298.15K_cal/(molK)", "S_298", "s_298") if c in df.columns]
        cand = legacy or [
            c for c in cols
            if not any(k in _norm(c) for k in _SKIP)
            and ("entropy" in _norm(c) or _re.match(r"^s0?(298|29815)", _norm(c)) or _norm(c) == "s")
        ]
        if len(cand) > 1:
            logger.warning(f"several S298 candidates {cand}; using {cand[0]!r} (override with --s_col)")
        s_col = cand[0] if cand else None
    # Cp(T)
    if cp_cols is None:
        cp_cols = []
        parsed = []
        for c in cols:
            m = _re.match(r"^\s*c_?p[^0-9]*?(\d+(?:\.\d+)?)", str(c).lower())
            if m and not any(k in _norm(c) for k in _SKIP):
                parsed.append((float(m.group(1)), c))
        for T in T_grid:
            hit = [c for t, c in parsed if abs(t - T) < 1.0]
            cp_cols.append(hit[0] if hit else None)
    else:
        if len(cp_cols) != len(T_grid):
            raise ValueError(f"--cp_cols has {len(cp_cols)} names but the T grid has {len(T_grid)} temperatures")
        cp_cols = [c if c not in ("", "none", "None") else None for c in cp_cols]
    for c in [h_col, s_col, *cp_cols]:
        _check(c)
    return {"h": h_col, "s": s_col, "cp": cp_cols}
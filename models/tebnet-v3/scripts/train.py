"""TEB-Net training / evaluation under the DeepTherm protocol
(Wang et al., "DeepTherm", Energy and AI 24 (2026) 100677; code: github.com/tairan-w/DeepTherm).

What is matched to DeepTherm
  * Data   : the 4,997-species curated set (Supplementary Data 1), SAME ROW ORDER (checked below).
  * Split  : exact port of DeepTherm's split_indices — 81 / 9 / 10 % train / val / test;
             random (np.random.default_rng(seed)), complexity (largest 10 % by heavy atoms = test),
             scaffold, functional-group hold-out (--test_smarts / --test_group).
  * Model selection : best checkpoint on the validation MAE averaged over the 9 DeepTherm targets
             (ΔH_f,298 in kcal/mol; S298 and Cp at 300…1500 K in cal/(mol·K)), or on the property
             subset given by --monitor. The full cosine LR schedule is run (no early stopping unless
             --patience is given; DeepTherm itself uses a constant LR with patience 15).
  * Metrics: test MAE / RMSE / R² per target in kcal/mol and cal/(mol·K), plus the per-heavy-atom
             and radical / closed-shell breakdowns used in the paper (see scripts/deeptherm_eval.py).
  * Ensemble (optional): --member m trains member m on the SAME test set (train ∪ val reshuffled);
             deeptherm_eval.py averages members with weights 1/val-MAE as in the paper.
What is NOT DeepTherm (it is our model): the architecture, QM9 pre-training (none) and optimiser.

Typical use
  # single models on 5 random splits (compare with the paper's single-model baselines, Fig. 6a–c)
  for s in 0 1 2 3 4; do python scripts/train.py --seed 0 --split_seed $s --split_type random; done
  # hierarchical complexity split (Fig. 6d–f)
  python scripts/train.py --seed 0 --split_type complexity
  # 10-member ensemble on split 0 (compare with DeepTherm itself)
  for m in $(seq 0 9); do python scripts/train.py --seed $m --member $m --split_seed 0; done
  python scripts/deeptherm_eval.py --runs benchmark_results_deeptherm
"""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import lightning.pytorch as pl
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_ROOT = PROJECT_ROOT.parent
TEBNET_MODEL_ROOT = BENCHMARK_ROOT.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TEBNET_MODEL_ROOT))
sys.path.insert(0, str(TEBNET_MODEL_ROOT / "data"))
from tebnet.data.dataset import ThermochemDataset
from tebnet.data.collate import ThermoCollater, HomologPairCollater
from tebnet.data.homologs import build_homolog_pairs, HomologPairDataset
from tebnet.data.splits import (
    deeptherm_split, ensemble_member_split, FUNCTIONAL_GROUP_SMARTS, DEEPTHERM_FRACTIONS,
)
from tebnet.data.baselines import fit_scalar_baseline, fit_scalar_baseline_fixed, fit_cp_cheb_baseline, R_GAS
from tebnet.models.tebnet import TEBNetV20
from tebnet.data.featurizer import (
    CleanBondFeaturizer,
    ThermoAtomFeaturizer,
    ThermoAtomFeaturizerV3,
    V15_NAMES,
    FEATURES_VERSION,
    v15_route_names,
    v15_base_names,
    ENTROPY_V2_NAMES,
    S_PRIOR_VARIANTS,
    s_prior_columns,
    entropy_v2_cache_path,
    load_or_compute_entropy_v2,
)
from tebnet.utils.callbacks import (
    ThermodynamicConsistencyCallback,
    PrintValidationMetricsCallback,
)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("tebnet_deeptherm")
KCAL_TO_J = 4184.0
CAL_TO_J = 4.184
J_TO_KCAL = 1.0 / KCAL_TO_J
J_TO_CAL = 1.0 / CAL_TO_J
DEEPTHERM_T_GRID = np.array([300.0, 400.0, 500.0, 600.0, 800.0, 1000.0, 1500.0])
COL = {n: i for i, n in enumerate(V15_NAMES)}
DEEPTHERM_SI_ORDER_SHA1 = "102f81bdb66cc6fd825d519880d10b4f84523e7b"
def order_fingerprint(h_j_per_mol: np.ndarray) -> str:
    s = ",".join(f"{round(float(v) * J_TO_KCAL, 2) + 0.0:.2f}" for v in h_j_per_mol)
    return hashlib.sha1(s.encode()).hexdigest()
def make_atom_featurizer(kind: str):
    if kind == "chemprop":
        return None
    if kind == "radical":
        return ThermoAtomFeaturizer.v2_radical()
    if kind == "thermo":
        return ThermoAtomFeaturizerV3.thermo()
    raise ValueError(kind)
def morgan_bits(mol, n_bits: int, radius: int = 2) -> np.ndarray:
    try:
        from rdkit.Chem import rdFingerprintGenerator
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        return gen.GetFingerprintAsNumPy(mol).astype(np.float32)
    except Exception:
        from rdkit.Chem import AllChem, DataStructs
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
class DeepThermValMAE(pl.Callback):
    """Logs val/deeptherm_mae: validation MAE averaged over the 9 DeepTherm targets
    (ΔH in kcal/mol, S and each Cp(T) in cal/(mol·K)) — DeepTherm's model-selection criterion.

    Computed here from its own forward pass on each validation batch (the model is in eval mode),
    so it does not depend on when Lightning aggregates the model's own val/* logs."""
    def __init__(self, n_temps: int):
        self.n_temps = n_temps
        self._reset()
    def _reset(self):
        self.sum = np.zeros(2 + self.n_temps)
        self.cnt = np.zeros(2 + self.n_temps)
    def on_validation_epoch_start(self, trainer, pl_module):
        self._reset()
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        with torch.no_grad():
            p = pl_module(batch)
        cols = [
            ((p["dh_f_298"] - batch.dh_f_298).abs() * J_TO_KCAL).unsqueeze(1),
            ((p["s_298"] - batch.s_298).abs() * J_TO_CAL).unsqueeze(1),
            (p["cp"] - batch.cp_grid).abs() * J_TO_CAL,
        ]
        err = torch.cat(cols, dim=1).detach().cpu().numpy().astype(np.float64)
        ok = np.isfinite(err)
        self.sum += np.where(ok, err, 0.0).sum(0)
        self.cnt += ok.sum(0)
    def on_validation_epoch_end(self, trainer, pl_module):
        per_target = np.where(self.cnt > 0, self.sum / np.maximum(self.cnt, 1), np.nan)
        value = float(np.nanmean(per_target)) if np.isfinite(per_target).any() else float("inf")
        pl_module.log("val/deeptherm_mae", value, prog_bar=True)
        pl_module.log("val/mae_H_kcal", float(per_target[0]))
        pl_module.log("val/mae_S_cal", float(per_target[1]))
        pl_module.log("val/mae_Cp300_cal", float(per_target[2]))
        pl_module.log("val/mae_SCp", float(np.nanmean(per_target[1:])))
def collect_predictions(model, loader, device):
    model.eval()
    out = {k: [] for k in ("h_pred", "s_pred", "cp_pred", "h_true", "s_true", "cp_true", "x")}
    smiles = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            p = model(batch)
            out["h_pred"].append(p["dh_f_298"].cpu().numpy())
            out["s_pred"].append(p["s_298"].cpu().numpy())
            out["cp_pred"].append(p["cp"].cpu().numpy())
            out["h_true"].append(batch.dh_f_298.cpu().numpy())
            out["s_true"].append(batch.s_298.cpu().numpy())
            out["cp_true"].append(batch.cp_grid.cpu().numpy())
            out["x"].append(batch.x_physic[:, : len(V15_NAMES)].cpu().numpy())
            smiles.extend(batch.smiles)
    res = {k: np.concatenate(v) for k, v in out.items()}
    res["smiles"] = smiles
    return res
def _stats(pred, true, mask=None) -> dict:
    m = np.isfinite(true) & np.isfinite(pred)
    if mask is not None:
        m &= mask
    if not m.any():
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "n": 0}
    e = pred[m] - true[m]
    den = np.square(true[m] - true[m].mean()).sum()
    return {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt(np.square(e).mean())),
            "r2": float(1 - np.square(e).sum() / den) if den > 0 else float("nan"), "n": int(m.sum())}
def deeptherm_metrics(p: dict, t_grid: np.ndarray, mask=None) -> dict:
    """Per-target metrics in DeepTherm units + the 9-target mean MAE (DeepTherm's criterion)."""
    res = {"H298_kcal": _stats(p["h_pred"] * J_TO_KCAL, p["h_true"] * J_TO_KCAL, mask),
           "S298_cal": _stats(p["s_pred"] * J_TO_CAL, p["s_true"] * J_TO_CAL, mask)}
    for j, T in enumerate(t_grid):
        res[f"Cp{T:.0f}_cal"] = _stats(p["cp_pred"][:, j] * J_TO_CAL, p["cp_true"][:, j] * J_TO_CAL, mask)
    cp_maes = [res[f"Cp{T:.0f}_cal"]["mae"] for T in t_grid]
    res["Cp_mean_MAE_cal"] = float(np.nanmean(cp_maes))
    res["mean9_MAE"] = float(np.nanmean([res["H298_kcal"]["mae"], res["S298_cal"]["mae"], *cp_maes]))
    return res
def subset_masks(x: np.ndarray) -> dict[str, np.ndarray]:
    rad = x[:, COL["n_rad_e"]] > 0
    nring = x[:, COL["n_rings"]]
    return {
        "all": np.ones(len(x), dtype=bool),
        "stable (closed-shell)": ~rad,
        "radicals": rad,
        "acyclic": nring == 0,
        "cyclic": nring > 0,
        "polycyclic": nring >= 2,
    }
def print_deeptherm_table(title: str, p: dict, t_grid: np.ndarray) -> dict:
    i300 = int(np.argmin(np.abs(t_grid - 300.0)))
    out = {}
    print(f"\n  {title}   (MAE; ΔH in kcal/mol, S & Cp in cal/(mol·K))")
    print(f"  {'subset':<22s} {'n':>5s} {'ΔH298':>8s} {'S298':>8s} {'Cp300':>8s} {'Cp mean':>8s} {'mean-9':>8s}")
    for name, m in subset_masks(p["x"]).items():
        if not m.any():
            continue
        r = deeptherm_metrics(p, t_grid, m)
        out[name] = r
        print(f"  {name:<22s} {int(m.sum()):>5d} {r['H298_kcal']['mae']:>8.3f} {r['S298_cal']['mae']:>8.3f} "
              f"{r[f'Cp{t_grid[i300]:.0f}_cal']['mae']:>8.3f} {r['Cp_mean_MAE_cal']:>8.3f} {r['mean9_MAE']:>8.3f}")
    return out
def main():
    ap = argparse.ArgumentParser(description="TEB-Net under the DeepTherm protocol")
    ap.add_argument("--seed", type=int, required=True, help="model-initialisation seed")
    ap.add_argument("--csv", type=str, default=str(PROJECT_ROOT / "deeptherm_data.csv"))
    ap.add_argument("--physics_cache", type=str, default="auto")
    ap.add_argument("--smiles_col", type=str, default="smiles")
    ap.add_argument("--h_col", type=str, default=None)
    ap.add_argument("--s_col", type=str, default=None)
    ap.add_argument("--cp_cols", type=str, default=None)
    ap.add_argument("--h_unit", choices=["kcal", "kj", "j"], default=None)
    ap.add_argument("--s_unit", choices=["cal", "j"], default=None)
    ap.add_argument("--cp_unit", choices=["cal", "j"], default=None)
    ap.add_argument("--max_epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=None,
                    help="early-stopping patience in epochs. Default None = no early stopping: the full cosine "
                         "LR schedule is run and the best-validation checkpoint is kept (chemprop's default). "
                         "Early stopping cuts off the low-LR tail of the cosine schedule, where most of the "
                         "fitting happens (random1 S/Cp run in v15.3: train S MAE 1.15 vs 0.50 on random0).")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--out_dir", type=str, default="benchmark_results_deeptherm")
    ap.add_argument("--split_type", choices=["random", "complexity", "size", "scaffold", "functional_group"],
                    default="random")
    ap.add_argument("--split_seed", type=int, default=0, help="seed of DeepTherm's split (repo default 0)")
    ap.add_argument("--split_fractions", type=float, nargs=3, default=list(DEEPTHERM_FRACTIONS),
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--test_smarts", type=str, default=None, help="functional_group split: SMARTS held out")
    ap.add_argument("--test_group", type=str, default=None, choices=sorted(FUNCTIONAL_GROUP_SMARTS),
                    help="functional_group split: one of DeepTherm's functional-group SMARTS")
    ap.add_argument("--member", type=int, default=0,
                    help="ensemble member: 0 = canonical split; m>0 = same test set, train∪val reshuffled")
    ap.add_argument("--monitor", choices=["deeptherm", "loss", "h", "s", "cp300", "scp"], default="deeptherm",
                    help="deeptherm = val MAE averaged over the 9 targets (DeepTherm criterion)")
    ap.add_argument("--atom_featurizer", choices=["chemprop", "radical", "thermo"], default="thermo")
    ap.add_argument("--baseline", choices=["none", "composition", "physics"], default="physics",
                    help="none = no frozen linear baseline (network predicts the full target around the train mean)")
    ap.add_argument("--loss_weights", type=float, nargs=3, default=[2.0, 1.5, 1.0], metavar=("H", "S", "CP"),
                    help="loss weights; e.g. '1 0 0' = ΔH-only model (use with --monitor h)")
    ap.add_argument("--no_3d", action="store_true")
    ap.add_argument("--s_prior", choices=list(S_PRIOR_VARIANTS), default="v1",
                    help="entropy prior: v1 = published v15 prior; v2_rot = exact S_trans + S_rot (lowest-energy "
                         "conformer) with fixed coefficient R, additive remainder fitted; see featurizer.s_prior_columns")
    ap.add_argument("--n_conf", type=int, default=10, help="ETKDG conformers per species for the v2 entropy terms")
    ap.add_argument("--conf_workers", type=int, default=8, help="processes for the v2 conformer search (cached)")
    ap.add_argument("--route_mode", choices=["grouped", "full"], default="grouped")
    ap.add_argument("--ridge_lambda", type=float, default=1e-3)
    ap.add_argument("--loss_norm", choices=["global", "residual"], default="global",
                    help="global = each target scaled by its train std (as DeepTherm/chemprop); "
                         "residual = scaled by the std left after the baseline (down-weights ΔH)")
    ap.add_argument("--weight_strategy", choices=["fixed", "uncertainty"], default="fixed")
    ap.add_argument("--cp_param", choices=["chebyshev", "monomial"], default="chebyshev")
    ap.add_argument("--ecfp_bits", type=int, default=0,
                    help="append a Morgan/ECFP4 bit vector to every branch input (DeepTherm uses 1024); 0 = off")
    ap.add_argument("--mp_depth", type=int, default=5)
    ap.add_argument("--mp_hidden", type=int, default=300)
    ap.add_argument("--branch_hidden", type=int, default=256)
    ap.add_argument("--branch_layers", type=int, default=2)
    ap.add_argument("--branch_dropout", type=float, default=0.1)
    ap.add_argument("--cp_branch_hidden", type=int, default=None)
    ap.add_argument("--cp_branch_dropout", type=float, default=None)
    ap.add_argument("--t_mid", type=float, default=1000.0)
    ap.add_argument("--t_high_dom", type=float, default=3000.0)
    ap.add_argument("--homolog_weight", type=float, default=0.0,
                    help="weight of the CH2-homolog increment penalty (0 = off, the default). The penalty is "
                         "already in units of the measured within-class scatter, so it needs no rescaling by "
                         "the data: 1e-3-1e-2 keeps it at roughly 5-15%% of the per-molecule loss, which is the "
                         "useful range for a prior. Larger values (>=0.1) make it comparable to or larger than "
                         "the main objective and the model stops fitting the targets.")
    ap.add_argument("--homolog_ramp_epochs", type=int, default=0,
                    help="linearly ramp the weight from 0 to --homolog_weight over this many epochs "
                         "(0 = switch on immediately). During warm-up the per-molecule loss is still moving "
                         "fast; ramping keeps the prior from dominating before the model has any signal.")
    ap.add_argument("--homolog_w_h", type=float, default=1.0, help="relative weight of ΔHf in the penalty")
    ap.add_argument("--homolog_w_s", type=float, default=1.0, help="relative weight of S298 in the penalty")
    ap.add_argument("--homolog_w_cp", type=float, default=1.0, help="relative weight of Cp in the penalty")
    ap.add_argument("--homolog_beta", type=float, default=1.0,
                    help="pseudo-Huber transition of the penalty, in tolerance units (1 = quadratic within "
                         "one measured scatter, linear beyond)")
    ap.add_argument("--homolog_cp_index", type=int, default=0,
                    help="index into the Cp temperature grid at which the Cp increment is imposed (0 = 300 K, "
                         "the only temperature the increment table can support)")
    ap.add_argument("--homolog_n_shrink", type=float, default=5.0,
                    help="shrinkage of a class increment towards the global increment (n/(n+n_shrink))")
    ap.add_argument("--homolog_tol_floor", type=float, default=0.5,
                    help="lower bound on a tolerance, as a fraction of the global increment scatter")
    ap.add_argument("--homolog_max_pairs_per_anchor", type=int, default=None,
                    help="cap the partners one anchor contributes (None = no cap)")
    ap.add_argument("--homolog_repeat", type=int, default=1,
                    help="emit each pair this many times per epoch (up-weights the term; the paired loader is "
                         "cycled up to the molecule loader length, so 1 already gives full coverage)")
    ap.add_argument("--tag_suffix", type=str, default="",
                    help="free label appended to the tag (e.g. _sigfix) to keep feature versions apart")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow re-running into a directory that already holds results.json")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    t_grid = DEEPTHERM_T_GRID
    t_mid = args.t_mid
    t_low_dom = float(t_grid[t_grid <= t_mid].min())
    use_3d = not args.no_3d
    if args.s_prior != "v1" and not use_3d:
        sys.exit("--s_prior v2*/v1_minconf needs 3-D features (drop --no_3d)")
    cache = None if args.physics_cache.lower() == "none" else args.physics_cache
    test_smarts = args.test_smarts or (FUNCTIONAL_GROUP_SMARTS[args.test_group] if args.test_group else None)
    tag = (f"{args.atom_featurizer}_{args.baseline}_{args.cp_param}_{'3d' if use_3d else '2d'}"
           f"_{args.route_mode}_d{args.mp_depth}_{args.loss_norm}"
           + (f"_ecfp{args.ecfp_bits}" if args.ecfp_bits else "")
           + ("_unc" if args.weight_strategy == "uncertainty" else "")
           + ("" if args.loss_weights == [2.0, 1.5, 1.0] else "_w" + "-".join(f"{w:g}" for w in args.loss_weights))
           + ("" if args.monitor == "deeptherm" else f"_mon-{args.monitor}")
           + ("" if args.s_prior == "v1" else f"_sp-{args.s_prior}"))
    short = {"mp_hidden": "mph", "branch_hidden": "bh", "branch_layers": "bl", "branch_dropout": "bd",
             "cp_branch_hidden": "cpbh", "cp_branch_dropout": "cpbd", "batch_size": "bs", "max_epochs": "ep",
             "patience": "pat", "ridge_lambda": "rl", "t_mid": "tmid", "t_high_dom": "thi"}
    for key, ab in short.items():
        val = getattr(args, key)
        if val != ap.get_default(key):
            tag += f"_{ab}{val:g}" if isinstance(val, float) else f"_{ab}{val}"
    if args.homolog_weight > 0.0:
        tag += f"_hom{args.homolog_weight:g}"
        if args.homolog_ramp_epochs:
            tag += f"_hramp{args.homolog_ramp_epochs}"
        if args.homolog_beta != ap.get_default("homolog_beta"):
            tag += f"_hbeta{args.homolog_beta:g}"
        w = [args.homolog_w_h, args.homolog_w_s, args.homolog_w_cp]
        if w != [1.0, 1.0, 1.0]:
            tag += "_hws" + "-".join(f"{x:g}" for x in w)
        if args.homolog_repeat != 1:
            tag += f"_hrep{args.homolog_repeat}"
    tag += args.tag_suffix
    split_name = args.split_type + (f"-{(args.test_group or test_smarts or '').replace(' ', '_')}"
                                    if args.split_type == "functional_group" else "") + f"{args.split_seed}"
    save_dir = str(PROJECT_ROOT / args.out_dir / tag / split_name / f"member{args.member}_seed{args.seed}")
    os.makedirs(save_dir, exist_ok=True)
    if (Path(save_dir) / "results.json").exists() and not args.overwrite:
        sys.exit(f"{save_dir} already has results.json — same configuration already run. "
                 f"Use --overwrite to replace it, or change --seed / --member.")
    lock = Path(save_dir) / "RUNNING.lock"
    if args.overwrite and lock.exists():
        lock.unlink(missing_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        sys.exit(f"another run is using {save_dir} (found {lock.name}). If no run is active, delete that file.")
    import atexit
    atexit.register(lambda: lock.unlink(missing_ok=True))
    logger.info(f"run directory: {save_dir}")
    full_dataset = ThermochemDataset.from_csv(
        args.csv, T_grid=t_grid, smiles_col=args.smiles_col, physics_cache=cache,
        h_col=args.h_col, s_col=args.s_col,
        cp_cols=[c.strip() for c in args.cp_cols.split(",")] if args.cp_cols else None,
        h_unit=args.h_unit, s_unit=args.s_unit, cp_unit=args.cp_unit,
        strict=True, compute_aux=False, compute_comp=False,
    )
    records = full_dataset.records
    smiles_list = [r["smiles"] for r in records]
    fp = order_fingerprint(np.array([r["dH_f_298"] for r in records]))
    same_order = (fp == DEEPTHERM_SI_ORDER_SHA1)
    logger.info(f"dataset: {len(records)} species | first SMILES {smiles_list[:3]} | "
                f"row order {'== DeepTherm SI (split reproduces the DeepTherm code)' if same_order else 'DIFFERS from DeepTherm SI'}")
    if not same_order:
        logger.warning("Row order / values differ from DeepTherm Supplementary Data 1 (expected 4,997 rows starting "
                       "COO, C=O, O=COO). The split is still DeepTherm's algorithm, but not the same test molecules.")
    tr0, va0, te0 = deeptherm_split(smiles_list, args.split_type, tuple(args.split_fractions),
                                    args.split_seed, test_smarts)
    tr_idx, val_idx, test_idx = ensemble_member_split(tr0, va0, te0, args.member, args.split_seed)
    logger.info(f"[split {split_name} | member {args.member}] {len(tr_idx)} train / {len(val_idx)} val / "
                f"{len(test_idx)} test")
    ds_kwargs = dict(T_grid=t_grid, aux_feature_set="v15", compute_aux=True, use_3d=use_3d)
    train_ds = ThermochemDataset([records[i] for i in tr_idx], **ds_kwargs)
    val_ds = ThermochemDataset([records[i] for i in val_idx], **ds_kwargs)
    test_ds = ThermochemDataset([records[i] for i in test_idx], **ds_kwargs)
    feature_names = list(V15_NAMES)
    use_s_v2 = args.s_prior != "v1" and args.baseline != "none"
    if use_s_v2:
        ent = load_or_compute_entropy_v2(smiles_list, entropy_v2_cache_path(args.csv), args.n_conf,
                                         workers=args.conf_workers)
        ent_of = {s: ent[i] for i, s in enumerate(smiles_list)}
        for ds in (train_ds, val_ds, test_ds):
            for dp in ds.datapoints:
                dp.x_physic = np.concatenate([dp.x_physic, ent_of[dp.smiles]]).astype(np.float32)
        feature_names += list(ENTROPY_V2_NAMES)
        logger.info(f"S prior {args.s_prior}: added {ENTROPY_V2_NAMES} "
                    f"({int(ent[:, -1].sum())} species without a valid conformer)")
    n_dense = len(feature_names)
    if args.ecfp_bits > 0:
        for ds in (train_ds, val_ds, test_ds):
            for dp in ds.datapoints:
                dp.x_physic = np.concatenate([dp.x_physic, morgan_bits(dp.mol, args.ecfp_bits)]).astype(np.float32)
        feature_names += [f"ecfp_{i}" for i in range(args.ecfp_bits)]
    atom_feat = make_atom_featurizer(args.atom_featurizer)
    collater = ThermoCollater(atom_featurizer=atom_feat)
    d_v = len(atom_feat) if atom_feat is not None else None
    dl = dict(batch_size=args.batch_size, collate_fn=collater, num_workers=args.num_workers,
              pin_memory=True, persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **dl)
    val_loader = DataLoader(val_ds, shuffle=False, **dl)
    test_loader = DataLoader(test_ds, shuffle=False, **dl)
    train_eval_loader = DataLoader(train_ds, shuffle=False, **dl)
    homolog_table = None
    homolog_loader = None
    if args.homolog_weight > 0.0:
        homolog_table = build_homolog_pairs(
            train_ds,
            n_shrink=args.homolog_n_shrink,
            tol_floor=args.homolog_tol_floor,
            cp_index=args.homolog_cp_index,
            max_pairs_per_anchor=args.homolog_max_pairs_per_anchor,
        )
        if len(homolog_table) == 0:
            logger.warning("--homolog_weight > 0 but the train split contains no CH2-homolog pairs: "
                           "the regulariser is disabled for this run")
        else:
            logger.info("CH2-homolog prior: " + homolog_table.summary())
            inc = homolog_table.increments.mean(axis=0)
            tol = homolog_table.tolerances.mean(axis=0)
            logger.info("  increments (shrunken, class median -> global, mean over classes): "
                        f"ΔHf {inc[0] / 4184.0:+.2f} kcal/mol | "
                        f"S {inc[1] / 4.184:+.2f} cal/(mol·K) | "
                        f"Cp {inc[2] / 4.184:+.2f} cal/(mol·K)")
            logger.info("  tolerances (mean over classes): "
                        f"ΔHf {tol[0] / 4184.0:.2f} kcal/mol | "
                        f"S {tol[1] / 4.184:.2f} | Cp {tol[2] / 4.184:.2f} cal/(mol·K)")
            homolog_ds = HomologPairDataset(train_ds, homolog_table, repeat=args.homolog_repeat)
            homolog_loader = DataLoader(
                homolog_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                collate_fn=HomologPairCollater(base=collater, table=homolog_table),
                num_workers=args.num_workers, pin_memory=True,
                persistent_workers=args.num_workers > 0,
            )
            logger.info(f"  paired loader: {len(homolog_ds)} items, {len(homolog_loader)} batches/epoch "
                        f"(molecule loader: {len(train_loader)})")
    train_combined = (CombinedLoader({"main": train_loader, "pairs": homolog_loader}, mode="max_size_cycle")
                      if homolog_loader is not None else train_loader)
    X = np.stack([dp.x_physic for dp in train_ds.datapoints]).astype(np.float64)
    H = np.array([dp.dh_f_298 for dp in train_ds.datapoints], dtype=np.float64)
    S = np.array([dp.s_298 for dp in train_ds.datapoints], dtype=np.float64)
    CP = np.stack([dp.cp_grid for dp in train_ds.datapoints]).astype(np.float64)
    if use_3d and X[:, COL["moi_valid"]].max() < 0.5:
        logger.warning("use_3d=True but no valid 3-D features (cache built with --no-3d?)")
    route = {k: v15_route_names(k, args.route_mode, use_3d) for k in ("h", "s", "cp")}
    if args.ecfp_bits > 0:
        ecfp_names = feature_names[n_dense:]
        route = {k: v + ecfp_names for k, v in route.items()}
    base = {k: v15_base_names(k, "composition" if args.baseline == "none" else args.baseline, use_3d)
            for k in ("h", "s", "cp")}
    s_fixed_R: dict[str, float] = {}
    if use_s_v2:
        base["s"], s_fixed_R = s_prior_columns(args.s_prior)
    fcol = {n: i for i, n in enumerate(feature_names)}
    cols = {k: [fcol[n] for n in v] for k, v in base.items()}
    Xb = np.zeros_like(X) if args.baseline == "none" else X
    bh = fit_scalar_baseline(Xb[:, cols["h"]], H, args.ridge_lambda)
    if s_fixed_R:
        fx = np.array([s_fixed_R.get(n, np.nan) * R_GAS for n in base["s"]])
        bs = fit_scalar_baseline_fixed(Xb[:, cols["s"]], S, fx, args.ridge_lambda)
    else:
        bs = fit_scalar_baseline(Xb[:, cols["s"]], S, args.ridge_lambda)
    bc = fit_cp_cheb_baseline(Xb[:, cols["cp"]], CP, t_grid, t_mid, t_low_dom, args.t_high_dom,
                              lam_rel=args.ridge_lambda)
    if args.cp_param == "monomial":
        bc["W"] = np.zeros_like(bc["W"])
        bc["b"] = np.zeros_like(bc["b"])
        bc["coef_std"] = np.ones_like(bc["coef_std"])
        bc["cp_resid_std"] = bc["cp_target_std"]
    logger.info(f"ΔH baseline: std {bh['target_std']*J_TO_KCAL:.2f} → {bh['resid_std']*J_TO_KCAL:.2f} kcal/mol | "
                f"S baseline: {bs['target_std']*J_TO_CAL:.2f} → {bs['resid_std']*J_TO_CAL:.2f} cal/(mol·K) | "
                f"Cp baseline residual {bc['cp_resid_std']*J_TO_CAL:.2f} cal/(mol·K)")
    coef_report = {}
    if args.baseline != "none":
        for n in base["h"]:
            if n.startswith(("rad_", "ring")) or n in ("n_OO1", "n_arom_rings", "n_bridgehead", "n_rad_e"):
                coef_report[f"H:{n}"] = float(bh["w"][base["h"].index(n)] * J_TO_KCAL)
        for n in ("log_sigma", "ln_mult", "is_chiral", "n_methyl", "log_I_rot", "S_trans_R", "S_rot_R",
                  "log_I_rot_min", "n_rotors"):
            if n in base["s"]:
                coef_report[f"S:{n} [R]"] = float(bs["w"][base["s"].index(n)] / R_GAS)
        logger.info("baseline coefficients (H: kcal/mol per count; S: units of R):\n  "
                    + "\n  ".join(f"{k:<22s}{v:+9.2f}" for k, v in coef_report.items()))
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd < 1e-3] = 1.0
    if args.ecfp_bits > 0:
        mu[n_dense:] = 0.0
        sd[n_dense:] = 1.0
    if args.loss_norm == "residual":
        ls_h, ls_s, ls_cp = bh["resid_std"], bs["resid_std"], bc["cp_resid_std"]
    else:
        ls_h, ls_s, ls_cp = bh["target_std"], bs["target_std"], bc["cp_target_std"]
    model = TEBNetV20(
        feature_names=feature_names,
        route_h=route["h"], route_s=route["s"], route_cp=route["cp"],
        base_h=base["h"], base_s=base["s"], base_cp=base["cp"],
        mp_hidden_dim=args.mp_hidden, mp_depth=args.mp_depth, mp_dropout=0.1,
        agg_type="sum", batch_norm=True, d_v=d_v, d_e=len(CleanBondFeaturizer()),
        branch_hidden=args.branch_hidden, branch_layers=args.branch_layers,
        branch_dropout=args.branch_dropout,
        cp_branch_hidden=args.cp_branch_hidden, cp_branch_dropout=args.cp_branch_dropout,
        t_ref=298.15, t_mid=t_mid, t_low_dom=t_low_dom, t_high_dom=args.t_high_dom,
        cp_param=args.cp_param,
        loss_type="huber",
        loss_weights={"h298": args.loss_weights[0], "s298": args.loss_weights[1], "cp": args.loss_weights[2]},
        weight_strategy=args.weight_strategy,
        loss_scale_cp=ls_cp, loss_scale_h=ls_h, loss_scale_s=ls_s,
        warmup_epochs=3, init_lr=1e-4, max_lr=8e-4, final_lr=1e-5, weight_decay=5e-5,
        homolog_weight=args.homolog_weight if (homolog_table is not None and len(homolog_table)) else 0.0,
        homolog_ramp_epochs=args.homolog_ramp_epochs,
        homolog_w_h=args.homolog_w_h, homolog_w_s=args.homolog_w_s, homolog_w_cp=args.homolog_w_cp,
        homolog_beta=args.homolog_beta, homolog_cp_index=args.homolog_cp_index,
    )
    model.set_baselines(bh, bs, bc)
    model.set_aux_norm(mu, sd)
    n_params = sum(p.numel() for p in model.parameters())
    monitor = {"deeptherm": "val/deeptherm_mae", "loss": "val/loss_total", "h": "val/mae_H_kcal",
               "s": "val/mae_S_cal", "cp300": "val/mae_Cp300_cal", "scp": "val/mae_SCp"}[args.monitor]
    callbacks = [
        DeepThermValMAE(n_temps=len(t_grid)),
        pl.callbacks.ModelCheckpoint(dirpath=save_dir, filename="best", monitor=monitor, mode="min", save_top_k=1),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        ThermodynamicConsistencyCallback(),
        PrintValidationMetricsCallback(),
    ]
    if args.patience is not None:
        callbacks.insert(2, pl.callbacks.EarlyStopping(monitor=monitor, patience=args.patience, mode="min",
                                                       verbose=True))
    trainer = pl.Trainer(
        max_epochs=args.max_epochs, accelerator="gpu", devices=1, precision="32-true",
        callbacks=callbacks, log_every_n_steps=10, default_root_dir=save_dir,
        gradient_clip_val=1.0, enable_progress_bar=False,
    )
    t0 = time.time()
    trainer.fit(model, train_combined, val_loader)
    t_train = time.time() - t0
    ckpt_cb = trainer.checkpoint_callback
    best_ckpt = ckpt_cb.best_model_path
    best_epoch = None
    if best_ckpt and Path(best_ckpt).exists():
        try:
            best_epoch = int(torch.load(best_ckpt, map_location="cpu", weights_only=False)["epoch"]) + 1
        except Exception:
            best_epoch = None
        model = TEBNetV20.load_from_checkpoint(best_ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    p_train = collect_predictions(model, train_eval_loader, device)
    p_val = collect_predictions(model, val_loader, device)
    p_test = collect_predictions(model, test_loader, device)
    val_m = deeptherm_metrics(p_val, t_grid)
    test_m = deeptherm_metrics(p_test, t_grid)
    train_m = deeptherm_metrics(p_train, t_grid)
    results = {
        "model": "TEB-Net", "tag": tag, "protocol": "deeptherm", "seed": args.seed, "member": args.member,
        "features_version": FEATURES_VERSION,
        "args": vars(args), "t_grid": t_grid.tolist(),
        "split": {"type": args.split_type, "seed": args.split_seed, "fractions": args.split_fractions,
                  "test_smarts": test_smarts, "same_order_as_deeptherm_SI": bool(same_order),
                  "order_sha1": fp},
        "n_train": len(tr_idx), "n_val": len(val_idx), "n_test": len(test_idx), "n_params": n_params,
        "epochs_trained": trainer.current_epoch, "best_epoch": best_epoch,
        "early_stopped": bool(trainer.current_epoch < args.max_epochs),
        "training_time_sec": round(t_train, 1),
        "best_val_monitor": float(ckpt_cb.best_model_score) if ckpt_cb.best_model_score is not None else None,
        "val_deeptherm_mae": val_m["mean9_MAE"],
        "baseline_coefficients": coef_report,
        "homolog": ({
            "enabled": True,
            "weight": args.homolog_weight,
            "ramp_epochs": args.homolog_ramp_epochs,
            "w_h": args.homolog_w_h, "w_s": args.homolog_w_s, "w_cp": args.homolog_w_cp,
            "beta": args.homolog_beta, "cp_index": args.homolog_cp_index,
            "n_shrink": args.homolog_n_shrink, "tol_floor": args.homolog_tol_floor,
            "repeat": args.homolog_repeat,
            "batches_per_epoch": len(homolog_loader),
            **homolog_table.to_json(),
        } if (homolog_table is not None and len(homolog_table)) else {"enabled": False}),
        "baselines": {"H_resid_std_kcal": bh["resid_std"] * J_TO_KCAL, "S_resid_std_cal": bs["resid_std"] * J_TO_CAL,
                      "Cp_resid_std_cal": bc["cp_resid_std"] * J_TO_CAL},
        "train": train_m, "val": val_m, "test": test_m,
        "checkpoint": best_ckpt,
        "test_indices": list(map(int, test_idx)), "val_indices": list(map(int, val_idx)),
        "train_indices": list(map(int, tr_idx)),
    }
    print("\n" + "=" * 86)
    print(f" TEB-Net | DeepTherm protocol | split {split_name} | member {args.member} | seed {args.seed}")
    print(f" {tag} | {n_params:,} params | {trainer.current_epoch} epochs (best {best_epoch}) | {t_train:.0f}s")
    print("=" * 86)
    results["test_subsets"] = print_deeptherm_table("TEST", p_test, t_grid)
    print_deeptherm_table("TRAIN", {k: v for k, v in p_train.items()}, t_grid)
    print("  per-T Cp test MAE: " + "  ".join(f"{T:.0f}K {test_m[f'Cp{T:.0f}_cal']['mae']:.3f}" for T in t_grid))
    print("=" * 86)
    with open(Path(save_dir) / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    np.savez(
        Path(save_dir) / "predictions.npz",
        **{f"test_{k}": v for k, v in p_test.items() if k != "smiles"},
        **{f"val_{k}": v for k, v in p_val.items() if k != "smiles"},
        test_smiles=np.array(p_test["smiles"], dtype=object),
        val_smiles=np.array(p_val["smiles"], dtype=object),
        test_indices=np.asarray(test_idx), t_grid=t_grid,
        feature_names=np.array(V15_NAMES, dtype=object),
    )
    print(f"  Results: {save_dir}/results.json\n")
if __name__ == "__main__":
    main()

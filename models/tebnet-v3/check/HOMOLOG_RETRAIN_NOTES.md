# CH2-homolog regulariser: retraining log and gotchas

Working notes for the retrain that is meant to test whether the CH2-homolog
consistency prior improves the manuscript's DeepTherm/homologous-series results.
Everything below is measured, not assumed; commands are given so each number can
be reproduced.

## 1. The batch-size bug (root cause of the failed first attempt)

The first full run (`benchmark_results_homolog/..._hom0.1`, launched 08:19) did not
learn at all: val ΔH298 MAE sat at ~22,700 J/mol for 82 epochs while the baseline
reached ~8,600 J/mol by epoch 10.

Diagnosis from the TensorBoard scalars, not from the loss value:

    python check/dump_homolog_series.py <run_dir>

At every logged step, `train/homolog_loss` was exactly **32x** the sum of the three
per-property diagnostics `train/homolog_pen_{h,s,cp}`:

    step   pen_h   pen_s   pen_cp  sum(pen)  homolog_loss   ratio
       9   1.468  0.05986  0.01262    1.5400       47.3215   30.7
      19   2.002  0.07428  0.02873    2.1054       67.2471   31.9
      29   2.421  0.07388  0.03804    2.5332       76.4950   30.2

32 is the training batch size (`--batch_size`, default 32). The property weights
`w` were broadcast against the already-reduced per-property means, producing a
`(B, 3)` tensor whose `sum()` counted each property B times, so the penalty grew
linearly with the batch size.

With `homolog_weight=0.1` the term then outweighed the main loss by 10-105x:

    python check/diagnose_homolog_scale.py <run_dir> --epochs 12

     ep         main      hom_pen        w        w*pen   pen/main    val_mae_H
      0       0.0571      59.8147      0.1       5.9815        105      23317.7
     10       0.0468       5.9959      0.1       0.5996       12.8      22856.5

The model optimised the *differences* between homologs (the penalty) and stopped
fitting the absolute values, which is exactly the frozen MAE above.

**Fixed** in `tebnet/nn/loss.py` (`per_prop` is reduced to `(3,)` before the weight
is applied) and guarded by `tests/test_homolog_collate.py::test_penalty_is_invariant_to_the_batch_size`.
The current code is batch-invariant:

    B=1    total=1998.000977  ratio(total/sum_per_prop)=1.00000
    B=32   total=1998.000977  ratio=1.00000
    B=128  total=1998.000488  ratio=1.00000

Runs started after 09:35 use the fixed code; the 08:19 run does not.

## 2. Choosing lambda

Once the reduction is right the penalty is O(1) while the main loss is
scale-normalised (`loss_scale_h` = 224,677 J/mol, so `0.5*(err/scale)^2` is ~7e-4
at MAE 8,600 J/mol). The penalty therefore still needs a small weight to sit at
5-15% of the objective. Measured penalty/main ratios:

| lambda | penalty/main (ep 0-5) |
|---|---|
| 1e-3 | 0.04 - 0.09 |
| 3e-3 | 0.09 - 0.19 |
| 0.1 (buggy run) | 10 - 105 |

## 3. Rebuilding the manuscript's homologous table

`check/homologous_controls.py --series <predictions.csv>` reproduces
`tab:homologous` / `tab:controls`.

The CSV that reproduces the paper numbers **exactly** is
`models/tebnet-v2/homologous_predictions.csv`:

| series | line | TEB-Net incr / MAE | paper |
|---|---|---|---|
| n-alkane | -4.79 | -4.99 / 0.46 | -4.99 / 0.46 |
| 1-alkyl radical | -4.84 | -4.95 / 0.28 | -4.95 / 0.28 |
| 1-alkyl hydroperoxide | -5.03 | -5.16 / 0.30 | -5.16 / 0.30 |
| 1-alkene | -4.92 | -5.04 / 0.24 | -5.04 / 0.24 |
| n-aldehyde | -4.89 | -5.22 / 2.47 | -5.22 / 2.47 |
| mean MAE | | 0.75 | 0.75 |

**Gotcha:** that file is dated 2026-10-05 16:39, while the v2 checkpoints it must
have come from are dated 18:00-19:44 the same day. The checkpoints were therefore
overwritten after the predictions were written, and re-running `predict_smiles.py`
on the surviving v2 checkpoints gives a different (much worse) answer:

    H, n-alkane: increment -2.93, MAE from line 10.59   (file: -4.99 / 0.46)

So the paper's homologous table cannot currently be regenerated from the
checkpoints on disk, and the v3 checkpoint trees give yet another answer:

| checkpoint set | n-alkane ΔH increment / MAE from line |
|---|---|
| v2 (on disk now) | -2.93 / 10.59 |
| v3 joint (10 members) | -5.35 / 2.77 |
| v3 decoupled splitH+splitSCp (10 members) | -3.88 / 4.28 |
| v2 predictions CSV (paper) | -4.99 / 0.46 |

Any new claim about the regulariser improving size extrapolation has to be made
against a baseline **re-evaluated with the same script in the same session**, not
against the numbers already in the .tex file.

## 3b. Final result of the lambda sweep (member 0, seed 0, split seed 0)

Both runs completed 2026-10-19 11:03 / 11:07, 100 epochs, no early stop.

| run | best epoch | val DeepTherm MAE (9 targets) | vs baseline |
|---|---|---|---|
| baseline `..._sp-v2_exact` | 84 | **0.9216** | — |
| `..._hom0.001` | 92 | 0.9991 | +8.4% worse |
| `..._hom0.003` | 92 | 1.0325 | +12.0% worse |

So on the aggregate validation metric the regulariser **hurts**: the penalty is
paying for consistency between homologs at the cost of absolute accuracy. This is
consistent with the per-epoch `val/mae_h298` trace (λ=1e-3 best 8091 J/mol vs
baseline ~7600-7900 J/mol).

The claim the regulariser is supposed to support is *size extrapolation*, not
aggregate MAE, so the deciding measurement is the homologous-increment table
(section 3) on these two checkpoints versus a baseline evaluated in the same
session. Do not quote the manuscript's 0.75 kcal/mol mean until that comparison
has been rerun, for the timestamp reason in section 3.

## 4. Commands

    # baseline homologous predictions (joint v3 ensemble, random split 0)
    H=benchmark_results_deeptherm/thermo_physics_chebyshev_3d_grouped_d5_global_sp-v2_exact/random0
    python scripts/predict_smiles.py --smiles_csv check/homologous_series.csv \
        --h_runs $H/member*_seed* --scp_runs $H/member*_seed* --out preds_baseline.csv

    # the same for a regularised run once it has finished (needs results.json)
    python scripts/predict_smiles.py --smiles_csv check/homologous_series.csv \
        --h_runs confirm_homolog/*hom0.001/random0/member*_seed* \
        --scp_runs confirm_homolog/*hom0.001/random0/member*_seed* --out preds_hom1e-3.csv

    # table + figure
    python check/homologous_controls.py --series preds_hom1e-3.csv
    python check/plot_homologous.py preds_hom1e-3.csv --out fig_homologous

    # per-member test/val MAE against the baseline
    python check/compare_homolog_run.py --baseline <baseline tree> --run <run tree>

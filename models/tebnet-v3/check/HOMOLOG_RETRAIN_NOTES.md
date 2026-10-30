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

## 3c. Same-session homologous comparison (the measurement that decides)

All four runs below were re-evaluated **in this session** with `check/predict_smiles.py`
on `check/homologous_series.csv`, so the numbers are comparable with each other even
though they are not comparable with the .tex file (section 3).

Errors beyond the DeepTherm set, per series (kcal/mol for ΔH):

| run | n-alkane | 1-alkyl radical | hydroperoxide | 1-alkene | n-aldehyde |
|---|---|---|---|---|---|
| baseline | 3.18 | 4.64 | 2.95 | 2.86 | 0.45 |
| λ=1e-3 | **1.91** | **0.53** | 3.15 | **1.77** | 5.69 |
| λ=3e-3 | **1.19** | **2.63** | **1.14** | **0.90** | 4.20 |
| λ=1e-3 + ramp 30 | 4.34 | 2.94 | 3.91 | 2.39 | 3.17 |

n-hexadecane (the single hardest extrapolation point, reference −89.6 kcal/mol):

| run | C16 prediction | error |
|---|---|---|
| baseline | −82.72 | 6.88 |
| λ=1e-3 | −90.85 | **1.25** |
| λ=3e-3 | −84.30 | 5.30 |
| λ=1e-3 + ramp 30 | — | 5.52 |

So the regulariser **does** do what it was introduced for: λ=1e-3 moves C16 from
6.9 kcal off to 1.3 kcal off (5.5x), and λ=3e-3 improves four of the five series.
The cost is absolute accuracy on the held-out test set:

| run | val ΔH | test ΔH | test S | test Cp300 |
|---|---|---|---|---|
| baseline | 1.606 | **1.692** | **1.510** | **0.936** |
| λ=1e-3 | 1.962 | 2.051 | 1.529 | 1.041 |
| λ=3e-3 | 2.156 | 2.379 | 1.555 | 1.152 |
| λ=1e-3 + ramp 30 | 1.824 | 1.945 | 1.544 | 1.054 |

The baseline test spread over the 10 members is s.d. 0.059 kcal (mean 1.625, range
1.531-1.703), so the +0.36 kcal of λ=1e-3 is ~7 s.d. — a real regression, not noise.

## 3d. The ramp hypothesis, and why it failed

Idea: switch the penalty on immediately (λ=1e-3, no ramp) and it fights the physics
branch from epoch 1, which is why absolute accuracy drops. Ramp the weight linearly
over 30 epochs so the model first learns the absolute values, then the increments.

Result: the ramp **does** recover most of the test accuracy (2.051 → 1.945) but it
gives the extrapolation back (C16 1.25 → 5.52, worse than every other regularised
run). Interpretation: the improvement in size extrapolation is produced by the penalty
acting during the whole fit, not by a short correction at the end. The ramp moves the
run back towards the baseline on both axes — it is a knob that interpolates between
"regulariser on" and "regulariser off", not a way to get both.

## 3e. THE DECIDING MEASUREMENT (2026-10-19, supersedes 3c/3d)

Section 3c compared **one** regularised member against the baseline **ensemble**, which
is not a like-for-like comparison. All ten baseline members of the decoupled-H tree
(`..._w1-0-0_mon-h_sp-v2_exact_splitH/random0`, the tree the manuscript's
`tab:homologous` is built from) were re-predicted in this session and each one scored
individually on n-hexadecane (NIST -89.60 kcal/mol):

| member | 1 | 2 | 3 | 4 | 5 | 6 | **7** | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| C16 err | 7.20 | 3.31 | 1.94 | 4.73 | **0.28** | 3.00 | **90.25** | 4.71 | **0.05** | - |
| increment | -4.13 | -5.04 | -4.94 | -4.28 | -4.75 | -4.72 | **+4.49** | -5.20 | -4.78 | - |

Over the ten members the C16 error is **7.02 +- 4.41 kcal (median 6.29, min 1.03,
max 16.69)** on the *joint* tree and, on the decoupled-H tree, members 5 and 9 already
land at 0.28 and 0.05 kcal -- i.e. **better than the best regularised run (1.25)**.

Consequences:

1. `lambda=1e-3` C16 = 1.25 is `z = -1.31` against the baseline member distribution.
   `lambda=3e-3` (5.30) is `z = -0.39`, ramp30 (11.44) is `z = +1.00`. **None of the
   regularised runs is statistically distinguishable from an unregularised member.** The
   "5.5x improvement" in 3c was single-member-versus-ensemble-mean, and the ensemble mean
   is inflated by one divergent member (see 3f).
2. The regulariser also does not fix the failure it was introduced for. On the
   decoupled tree it is *member 7* that diverges (+4.49 kcal/mol per CH2), and the
   regularised runs were trained on the joint tree, so the divergent member is still
   there. Section 3c's "the regulariser makes C16 extrapolate" is therefore not
   supported: the baseline already contains members that extrapolate perfectly.
3. Absolute accuracy is still worse: best val dH298 over 100 epochs is baseline
   **6500 J/mol** (ep 59) vs 7988 (1e-3), 8817 (3e-3), 7633 (ramp30), and the two
   lower-weight runs still running (5e-4, 3e-4) sit at ~7000-7100 J/mol, i.e. *not*
   better than 6500 either.

**Verdict:** the CH2-homolog regulariser, at every weight tested (3e-4 ... 3e-3, with and
without a ramp), does not beat the unregularised baseline on either axis once the
comparison is member-to-member. It should be reported as a negative/ablated result, not
as a headline contribution.

### 3e-bis. Complete lambda sweep, 100 epochs, member 0, seed 0, split 0

All rows are the best-epoch value over the full 100-epoch cosine schedule; no early stop.

| lambda | best val dH298 (J/mol) | best epoch | val 9-target | test dH (kcal) | test S | test Cp300 |
|---|---|---|---|---|---|---|
| 0 (baseline) | **6500** | 59 | **0.9216** | **1.692** | **1.510** | **0.936** |
| 3e-4 | 7087 | 67 | 0.9832 | 1.753 | 1.514 | 0.973 |
| 5e-4 | 7098 | 70 | 0.9547 | 1.806 | 1.523 | 1.008 |
| 1e-3 | 7988 | 82 | 0.9991 | 2.051 | 1.529 | 1.041 |
| 3e-3 | 8817 | 49 | 1.0325 | 2.379 | 1.555 | 1.152 |
| 1e-3 + ramp 30 | 7633 | 91 | 0.9706 | 1.945 | 1.544 | 1.054 |

The trade-off is monotone in lambda and the baseline is the best point on **every**
column. Lowering lambda towards zero only interpolates back to the baseline, so no
single lambda "hits both axes": at 3e-4 the penalty is already too weak to change the
extrapolation, and by 1e-3 the test error has grown by 0.36 kcal (~6 sd of the
10-member spread, 0.059). This closes the sweep: the regulariser is a **negative
result** and must not be presented as a contribution.

## 3f. The real result: robust aggregation removes the divergent member for free

The same per-member predictions give a much stronger and fully supported claim. On the
45 out-of-set homologues and on C16 (decoupled 10-member ensemble, random0):

| aggregation | C16 error | mean MAE from the line |
|---|---|---|
| weighted mean (manuscript) | 10.64 | 3.01 |
| simple mean | 10.55 | 2.99 |
| **median** | **0.05** | **1.21** |
| trimmed mean (drop min/max) | 1.35 | 0.83 |
| drop member 7 by hand (manuscript) | 0.52 | -- |

So the manuscript's own remedy (hand-pruning member 7 after inspecting the series) is
unnecessary: a **median over ensemble members** gets C16 to 0.05 kcal with no
out-of-sample information, no label-free heuristic and no hand-picked member. This is a
strictly better story than "we found the outlier and removed it", because median
aggregation is the standard robust estimator and generalises to members that have not
been inspected.

Also note the manuscript's claim that validation MAE cannot flag the divergent member is
correct and now quantified: Spearman(val dH MAE, C16 error) = **+0.067, p = 0.865**
(n = 10), and member 7 ranks 4/10 on validation. Robust aggregation is the answer, not
model selection.

## 3g. Provenance checks (all three resolved; one earlier suspicion retracted)

* **Retraction:** I first suspected that `tab:deeptherm`'s "mean +- s.d. over split seeds
  0-4" was really ten training seeds on split 0. That is **wrong** and the manuscript is
  correct. Recomputed over the five split seeds (one model each): H 1.409 +- 0.159,
  S 1.542 +- 0.074, Cp 0.881 +- 0.048 -- exactly the paper's 1.41 +- 0.16 / 1.55 +- 0.07
  / 0.89 +- 0.05. Do not "fix" this footnote.
* `tab:deeptherm` ensemble values also reproduce: weighted ensemble of the decoupled
  trees gives H 1.273 / S 1.420 / Cp 0.833 (paper: 1.28 / 1.42 / 0.83).
* `tab:homologous` and `tab:controls` reproduce exactly, including 2.73 mean MAE and
  9.73 kcal for C16 -- see `check/` session output.
* `member0_seed0` of the decoupled-H tree has a **stale `predictions.npz`** (1500 test
  rows instead of 501). Any ensemble rebuilt from `.npz` must drop it or align on
  `test_indices`; the numbers above were checked both ways and agree.

## 3h. Recommended aggregators (verified, ready to paste)

Ten-member decoupled trees, random split 0, 45 out-of-set homologues:

| aggregator | increment range | mean MAE from lines | C16 err | worst homologue | n with err > 5 kcal |
|---|---|---|---|---|---|
| weighted mean (current) | -3.88 .. -4.34 | 2.73 | 9.73 | 7.56 | 8 |
| median | -4.96 .. -5.29 | 1.05 | 0.34 | 4.41 | 0 |
| **trimmed mean** | **-4.85 .. -5.14** | **0.75** | **1.26** | **3.25** | **0** |
| lines (reference) | -4.79 .. -5.03 | 0 | -- | -- | -- |

In-distribution cost is nil (501 test species, dH): weighted mean 1.2729, median 1.2643,
trimmed 1.2633; S and Cp300 differ by <= 0.002. Leave-one-member-out stability of the
mean MAE: weighted mean 2.85 (worst 3.28), median 1.10 (worst 1.22). Trimmed mean on
random subsets: k=8 gives median C16 err 1.22, max 2.56 over 500 draws.

`S` is unaffected by the choice (6.31 / 7.05 / 6.34), so the robust aggregator is not a
universal fix -- it targets the *increment* failure, which is a dH phenomenon here.

## 3i. Provenance audit of the robust-aggregation numbers (2026-10-19, second pass)

Every number in §3.2, Table 5 and Tables S9 / S10 was recomputed from the stored
per-member predictions with `check/aggregation_report.py` (per-member values are in
`check/homologous_members.csv`). **Seven defects were found and are fixed in the
manuscript; five numbers change.**

### (i) `member0_seed0` of the decoupled-H tree is not on the 81/9/10 split

Its `results.json` records `fractions = (0.6, 0.1, 0.3)`, `n_train = 2998`,
`n_test = 1500`, whereas members 1-9 record `(0.81, 0.09, 0.1)`, `n_train = 4047`,
`n_test = 501`. The 501 test indices of member 0 are a subset of its 1500, so it is a
*separate run*, not a member of the 81/9/10 ensemble. Consequences:

* the 10-member in-distribution ensemble mixes two splits; the 9-member one is the
  ensemble the manuscript describes. Both round to the published Table 1 values, so
  Table 1 is unaffected (H 1.2798 vs 1.2729, S 1.4198 vs 1.4202, Cp300 0.8327 vs 0.8329).
* the numbers 1.2729 / 1.2633 / 1.2643 quoted in §3.2 and in Table 5 footnote *a* are the
  **nine-member** values (the script that produced them skipped member 0 because its
  `.npz` has 1500 rows). They were labelled "ten-member". Fixed.

### (ii) The Spearman correlation was computed on nine members, not ten

`rho = +0.067, p = 0.865` is what you get for **n = 9** (member 0 excluded). For the ten
members it is `rho = -0.139, p = 0.701`. The manuscript claimed `+0.07, p = 0.87, n = 10`,
which is a mix of the two. The claim itself ("validation MAE does not flag the divergent
member") is unaffected -- a correlation of -0.14 with p = 0.70 is the same absence of
signal -- and member 7 ranks 4/10 on validation either way. Fixed to the n = 10 values.

### (iii) The divergent member's increment is not +4.49

Member 7 gives `+6.13, +2.93, +3.97, +4.45, +3.27` kcal/mol per CH2 over the five series.
No natural summary equals 4.49: mean +4.15, median +3.97, pooled slope +4.68,
slope of (member 7 - line) +8.85. The manuscript quoted "+4.49 per CH2". Fixed to the
measured range +2.93 to +6.13 (already what Table 5 and Table S10 print).

### (iv) The in-distribution S / Cp tolerance was understated

With ten members the trimmed mean moves S by +0.0013 and Cp(300 K) by +0.0027
cal mol^-1 K^-1, not "within 0.002". Fixed to "within 0.003".

### (v) Table S10: the C16 error of the no-prior control was wrong

The row read 11.83 kcal; `scratch/no_prior_h_v3.csv` (the file whose per-series increments
match footnote *d* exactly: -3.27, -2.62, -3.24, -3.60, -3.08) gives a C16 prediction of
-70.76 kcal, i.e. an error of **18.84** kcal. Fixed.

### (vi) Table S10 footnotes were out of order

They ran a, b, c, d, f, e. Renumbered to a, b, c, d, e, f.

### (vii) Fig. 3 drew only the weighted mean

The figure therefore contradicted the section it illustrates. `check/plot_homologous.py`
now takes `--aggregation {trimmed,weighted}` (default `trimmed`) and draws the trimmed mean
solid, the weighted mean thin, and the two controls named in the caption. The underlying
numbers are unchanged; `homologous_predictions.csv` gained `trim_H_kcal`, `trim_S_cal` and
`trim_Cp300_cal`, written by `predict_smiles.py`.

### Numbers that do reproduce exactly

* Table 5: weighted mean 2.73 / 9.73; trimmed mean 0.75 / 1.26; median 1.05 / 0.34;
  increments -3.88..-4.34, -4.85..-5.14, -4.96..-5.29; worst homologue 7.56 / 3.25 / 4.41.
* Table S10: every member row, including member 7 (31.17 / 90.25) and the nine-member
  summary (1.86, 0.75-3.09; 2.87, 0.05-7.20).
* Table S9: S increments 7.04-7.76 and Cp 5.41-5.50; S MAE 6.32 / 6.34 / 7.05 and
  Cp MAE 1.23 / 1.23 / 1.27 for weighted / trimmed / median. The entropy deficit is
  unchanged by the aggregator (6.32 -> 6.34), as the Discussion states.
* DeepTherm: 11.26 mean MAE from the lines (44 homologues), C16 28.47 kcal, 39.44 cal/mol/K.
* k-subset draws: k=8 max C16 error 2.56 over 500 draws (identical for five RNG seeds),
  k=9 max 1.91 (exhaustive, ten leave-one-out subsets).
* Leave-one-member-out stability of the mean MAE: weighted 2.85, trimmed 1.10, median 1.10.

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

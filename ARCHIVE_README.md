# TEB-Net research tree — archive notes

Snapshot taken 2026-10-19 of `/home/nvsinh1/brats_segmentation/teb-net-model`,
with the AI-agent scaffolding removed. Everything needed to reproduce the
DeepTherm results and the manuscript tables is included.

## Layout

| path | what it is |
|---|---|
| `models/tebnet-v3/` | **active** TEB-Net. `tebnet/` package, `scripts/train.py`, `scripts/predict_smiles.py`, `scripts/deeptherm_eval.py`, `check/` (manuscript numbers + analysis scripts), `tests/` (58 tests). |
| `models/tebnet-v2/` | previous model, kept because the manuscript's homologous-series table was produced with it (`homologous_predictions.csv`). |
| `models-baseline/` | DeepTherm protocol datasets, SOTA reference implementations, baseline results. |
| `framework/` | `chemistry_framework` package + benchmarks (separate project, `soratori1025/Framework-Chemistry`). |
| `data/`, `figures/` | shared data and manuscript figures. |

## What was removed and why

The `framework/` directory had been populated by ~50 different AI coding agents
(`.claude/`, `.roo/`, `.augment/`, `.qwen/`, `.windsurf/`, `.trae/`, …). Each of
those directories contained nothing but a `skills/` subdirectory holding 177
symlinks into `.agents/skills/`. No research code referenced them
(`grep -rn skills framework/chemistry_framework framework/benchmarks framework/tests`
returned nothing), they were untracked in git, and the whole set is reproducible
from `skills-lock.json` (`K-Dense-AI/scientific-agent-skills`).

They were **moved, not deleted**, to
`/home/nvsinh1/quarantine_tebnet_2026-10-19/agent_skills/` (27 MB) so the removal
is reversible. `framework/skills/` and `skills-lock.json` went there too.

Also excluded from this archive (not deleted from the server):

| excluded | size | how to restore |
|---|---|---|
| `.venv/` | 6.8 G | `python -m venv .venv && .venv/bin/pip install -e models/tebnet-v3` |
| `models-baseline/sota/*/.git/` | 2.3 G | the four repos are public: `chemprop/chemprop`, `yureel/HC-Thermochemistry`, `tairan-w/DeepTherm` |
| `framework/.git/` | 30 M | `git clone https://github.com/soratori1025/Framework-Chemistry` (0 commits ahead of origin) |
| `__pycache__/`, `.pytest_cache/`, `*.pyc` | small | regenerated automatically |

Every checkpoint, `results.json`, `predictions.npz`, dataset CSV, `check/` script
and figure **is** in the archive.

## Reproducing

```bash
python -m venv .venv && .venv/bin/pip install -e models/tebnet-v3
cd models/tebnet-v3
.venv/bin/python -m pytest tests/ -q          # 58 passed

# baseline (no regulariser)
.venv/bin/python scripts/train.py --csv deeptherm_data.csv --s_prior v2_exact --gpu 0

# with the CH2-homolog regulariser
.venv/bin/python scripts/train.py --csv deeptherm_data.csv --s_prior v2_exact \
    --homolog_weight 1e-3 --gpu 0

# manuscript tables
.venv/bin/python check/paper_numbers.py
.venv/bin/python check/homologous_controls.py --series homologous_predictions.csv
```

## Read this before quoting the homologous-series numbers

`models/tebnet-v3/check/HOMOLOG_RETRAIN_NOTES.md` records two findings that
affect any claim about the CH2-homolog regulariser:

1. The first regularised run diverged because the penalty was multiplied by the
   batch size (a `(B,3)` tensor was summed instead of a `(3,)` one). Fixed and
   regression-tested; only runs after 2026-10-19 09:35 are valid.
2. The manuscript's homologous table cannot be regenerated from the v2
   checkpoints on disk — `tebnet-v2/homologous_predictions.csv` is timestamped
   *before* the checkpoints it must have come from, and re-running
   `predict_smiles.py` on those checkpoints gives different numbers
   (−2.93 / 10.59 instead of −4.99 / 0.46 for n-alkane).

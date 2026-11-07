# MantisV2: reproduction and improvements

COMP41850 AI for Time Series, group project (UCD, Autumn 2026).

We reproduce the zero-shot results of **MantisV2** (Feofanov et al., 2026, arXiv:2602.17868) and then improve the
model with three changes on top of the pretrained encoder:

1. **Cross-channel mixer.** MantisV2 encodes every channel separately. We add a small, gated attention layer across channels.
2. **Learned token pooling.** The paper uses only layer 2 (CLS + mean token). We learn a weighted mix of all 6 layers
   and attention pooling over the tokens.
3. **LoRA fine-tuning.** Low-rank adapters (rank 8) on the attention weights. This trains 36,864 encoder weights (0.9% of 4.2M).

**Status (7 Nov 2026): all experiments are done.** 17 of the 18 datasets in Tables 2 and 3 of the paper are run with
every method; Ego4D is not run because it needs a signed licence agreement. Only the report remains.
Full tables (with std, and the mixer gate for every dataset): **[`results/RESULTS.md`](results/RESULTS.md)**.

## Main results (accuracy, mean of 3 seeds)

**Reproduction.** Frozen MantisV2 + Random Forest matches the paper on 15 of 17 datasets to within 0.01
(16 within 0.02; mean absolute difference 0.004). The two larger gaps are MP8 (−0.010) and MP50 (−0.025); CAP-ID is
−0.006 (the paper does not report the seed of its random split).

| # | Dataset | Paper | Our RF | Mixer v2 (idea 1) | Idea 2 + mixer | LoRA + mixer (idea 3) | Best overall |
|---|---|---|---|---|---|---|---|
| 1 | Ego4D | 0.526 | not run (licence) | | | | |
| 2 | EMOPain | 0.880 | 0.883 | 0.854 | 0.839 | 0.864 | RF 0.883 |
| 3 | HHAR-ID | 0.985 | 0.983 | 0.989 | 0.989 | **0.990** | LoRA |
| 4 | HHAR-OOD | 0.582 | 0.587 | 0.520 | 0.540 | 0.570 | RF 0.587 |
| 5 | MP8 | 0.686 | 0.676 | **0.711** | 0.629 | 0.695 | mixer |
| 6 | MP50 | 0.735 | 0.709 | 0.713 | 0.693 | **0.765** | LoRA |
| 7 | UCI-HAR | 0.901 | 0.901 | 0.918 | 0.913 | **0.925** | LoRA |
| 8 | Blink | 0.996 | 0.996 | 0.998 | 0.992 | **1.000** | LoRA |
| 9 | CAP-ID | 0.815 | 0.809 | 0.826 | **0.834** | 0.819 | idea 2 + mixer |
| 10 | CAP-OOD | 0.786 | 0.786 | 0.768 | 0.765 | 0.761 | RF 0.786 |
| 11 | Epilepsy-EEG | 0.956 | 0.956 | 0.958 | 0.958 | 0.958 | idea 2 pooling 0.959 |
| 12 | FingerMovements | 0.550 | 0.550 | 0.543 | 0.550 | 0.597 | LogReg 0.640 |
| 13 | PCL-ID | 0.572 | 0.577 | 0.599 | 0.597 | **0.620** | LoRA |
| 14 | PCL-OOD | 0.531 | 0.534 | **0.562** | 0.559 | 0.558 | mixer |
| 15 | SEDFx-ID | 0.800 | 0.799 | 0.814 | **0.814** | 0.813 | idea 2 + mixer |
| 16 | SEDFx-OOD | 0.764 | 0.764 | **0.798** | 0.796 | 0.795 | mixer |
| 17 | SelfRegulationSCP1 | 0.813 | 0.813 | 0.832 | 0.795 | 0.854 | LogReg 0.867 |
| 18 | SelfRegulationSCP2 | 0.517 | 0.515 | 0.494 | **0.535** | 0.515 | idea 2 + mixer |

**Over the 17 datasets:**

| Method | Mean accuracy | Beats RF on | Best method on |
|---|---|---|---|
| RF (paper protocol) | 0.755 | — | 3 datasets |
| Mixer v2 (idea 1) | 0.758 | 12 of 17 | 3 |
| Idea 2 + mixer | 0.753 | 9 of 17 | 3 |
| LoRA + mixer (idea 3) | **0.771** | **13 of 17** | **5** |

Key findings for the report:

- **LoRA + mixer is the strongest improvement**: +1.5 points over the paper's protocol on average, training only
  36,864 encoder weights (0.9% of the model).
- **Biggest gains over the paper's protocol**: MP50 +5.6 points (LoRA), PCL-ID +4.3 (LoRA), SEDFx-OOD +3.4 (mixer),
  MP8 +3.5 (mixer), PCL-OOD +2.7 (mixer), CAP-ID +2.5 (idea 2 + mixer). The paper's Random Forest stays best on
  HHAR-OOD, CAP-OOD and EMOPain: two of these are out-of-domain tests, where the trained heads fit the training
  domains too closely.
- **The mixer gate shows how much the model uses channel mixing**: |gate| about 0.05–0.3 on multi-channel data
  (largest on HHAR, MP8, CAP, SEDFx), and close to 0 on the single-channel Epilepsy-EEG and on the small EEG sets
  FingerMovements and SelfRegulationSCP1/2.
- **Learned layer weights stay close to uniform** (0.14–0.22 per layer; a slight preference for the last two layers
  on EMOPain and MP8), so no single layer dominates; this questions the paper's fixed choice of layer 2.
- **Small EEG datasets (FingerMovements, SCP1) are best with plain logistic regression**: with a few hundred training
  samples a trained head overfits.

Notes: LoRA on large datasets (more than 10,000 training samples) uses 4 epochs to fit the free GPU. For datasets with
more than about 40 channels, the LoRA batch is reduced to fit GPU memory (see `kit/mantisv2_kit.py`, `fit`).

## Run it on other datasets

Everything is in **[`kit/`](kit/)**; the guide is **[`kit/HOW_TO_RUN.md`](kit/HOW_TO_RUN.md)**, the per-dataset sources
are in **[`kit/DATASETS_1-18.md`](kit/DATASETS_1-18.md)**, and `kit/run_datasets_1-6_13-18_kaggle.py` is the exact
Kaggle runner used for datasets 1–6 and 13–18. `python kit/make_results_tables.py` rebuilds `results/RESULTS.md` from
all result CSVs.

## Repository layout

```
kit/                           ready-to-use version of everything, for any dataset
  mantisv2_kit.py              all methods + data loaders + training (also a command-line tool)
  MantisV2_Kit_Kaggle.ipynb    upload to Kaggle/Colab, edit one cell, run
  HOW_TO_RUN.md                step-by-step guide
  DATASETS_1-18.md             the 18 datasets, who runs which, how to load each
  combine_results.py           merges everyone's CSVs into one table
  make_notebook.py             rebuilds the notebook after changes to mantisv2_kit.py
  run_datasets_1-6_13-18_kaggle.py   the Kaggle runner used for datasets 1-6 and 13-18 (parts 1-5)
  make_results_tables.py       builds results/RESULTS.md from every result CSV
reproduction/
  reproduce_mantisv2.py        standalone script: frozen MantisV2 + Random Forest, datasets 7-12
  MantisV2_Reproduction_*.ipynb/.html/.pdf   executed notebook (Kaggle run)
  REPRODUCTION_NOTES.md        protocol, data sources and problems we hit
improvements/
  code/                        notebook cells for each run, one "# %% cell" block per notebook cell
    1_ideas_1_2_first_try.py   first version of the mixer + learned pooling (overfit; kept for the report)
    2_mixer_v2.py              fixed mixer: small, gated, channel dropout
    3_idea2_cap_and_lora.py    learned pooling on CAP + LoRA on all six datasets
  runs/                        executed notebooks (.ipynb/.html/.pdf) and CSV for each run
    4_datasets_1-6_13-18/      Kaggle runs for datasets 1-6 and 13-18 (part 2 ran out of RAM; re-run in part 5)
results/
  RESULTS.md                   final tables for all 18 datasets
  datasets_7_12_kit_results.csv   datasets 7-12 in the kit's format
  datasets_1-18_kit/           kit-format CSVs from the runs above
```

## How to run

Reproduction (CPU is enough except for CAP, which is slow):

```bash
pip install -r requirements.txt
python reproduction/reproduce_mantisv2.py --datasets Blink FingerMovements
```

Datasets are downloaded automatically into `data/` (not stored in git). CAP is an 8.7 GB download.

Improvements: we ran these on Kaggle (free GPU T4). Make a new Kaggle notebook, turn on the GPU, and paste each
`# %% cell` block from `improvements/code/` into its own cell. Then use "Save & Run All". Run 3 needs about 70
minutes on a T4.

## Data sources

- Blink, FingerMovements: timeseriesclassification.com
- UCI-HAR, Epilepsy-EEG: TF-C preprocessed files on figshare (articles 19930244, 19930199), fetched with the figshare API
- CAP, HHAR, PCL, SEDFx: WOODS preprocessed `.h5` files (Google Drive ids in `kit/DATASETS_1-18.md`)
- EMOPain, SelfRegulationSCP1/2: timeseriesclassification.com
- MP8, MP50: Sungu Isiacik et al. (2025), github.com/mlgig/FM4MTSC (public Google Drive folder)
- Ego4D: not used (licence agreement required)

Pretrained model: `paris-noah/MantisV2` from the `mantis-tsfm` package (github.com/vfeofanov/mantis).

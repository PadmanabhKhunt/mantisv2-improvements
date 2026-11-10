# MantisV2: reproduction and improvements

COMP41850 AI for Time Series, group project (UCD, Autumn 2026).

We reproduce the zero-shot results of **MantisV2** (Feofanov et al., 2026, arXiv:2602.17868) and then improve the
model with three changes on top of the pretrained encoder:

1. **Cross-channel mixer.** MantisV2 encodes every channel separately. We add a small, gated attention layer across channels.
2. **Learned token pooling.** The paper uses only layer 2 (CLS + mean token). We learn a weighted mix of all 6 layers
   and attention pooling over the tokens.
3. **LoRA fine-tuning.** Low-rank adapters (rank 8) on the attention weights. This trains 36,864 encoder weights (0.9% of 4.2M).

**Status (7 Nov 2026): all experiments are done.** 17 of the 18 datasets in Tables 2 and 3 of the paper are run with
every method; Ego4D is not run because it needs a signed licence agreement.

**Where to look first**

| You want | Open |
|---|---|
| The report (IEEE format) | [`report/main.pdf`](report/main.pdf) |
| All final numbers, one row per dataset × method | [`results/all_results.csv`](results/all_results.csv) (or one file per dataset in [`results/by_dataset/`](results/by_dataset/)) |
| The same numbers as readable tables (with std and mixer gates) | [`results/RESULTS.md`](results/RESULTS.md) |
| How long every run and every step took | [`results/TIMING.md`](results/TIMING.md) |
| The executed Kaggle notebook behind every number | [`kaggle_runs/`](kaggle_runs/) (numbered 01–10, with a table of what each did) |
| The code | [`src/`](src/) |
| How to run it, the 18 datasets, reproduction notes | [`docs/`](docs/) |

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

Key findings:

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
more than about 40 channels, the LoRA batch is reduced to fit GPU memory (see `src/mantisv2_kit.py`, `fit`).

## Repository layout

```
README.md                      this file: overview, main results, how to run
requirements.txt
src/                           all code
  mantisv2_kit.py              the kit: data loaders + all 7 methods (RF, LogReg, head, mixer, pool, pool+mixer, LoRA)
                               for any dataset; also a command-line tool
  MantisV2_Kit_Kaggle.ipynb    the kit as a ready-to-run Kaggle/Colab notebook (built by make_notebook.py)
  make_notebook.py             rebuilds the notebook after changes to mantisv2_kit.py
  reproduce_mantisv2.py        standalone reproduction of the paper protocol (frozen MantisV2 + Random Forest)
  run_kaggle_parts.py          the Kaggle runner used for runs 05-09 (PART = 1..5)
  development/                 the scripts of runs 02-04, written while developing the three ideas
    1_ideas_1_2_first_try.py   first mixer + learned pooling (they overfit; kept for the report)
    2_mixer_v2.py              the fixed mixer: small, gated, channel dropout
    3_idea2_cap_and_lora.py    learned pooling on CAP + LoRA
  tables/                      turn the raw run outputs into the final tables and figures
    build_all_results.py       kaggle_runs/*  ->  results/all_results.csv, results/by_dataset/
    make_results_tables.py     -> results/RESULTS.md
    make_timing_table.py       -> results/TIMING.md and the report's timing tables
    make_figures.py            -> report/figures/
    combine_results.py         quick dataset x method table from any folder of kit CSVs
    common.py                  dataset list, paper numbers, method names
kaggle_runs/                   the 10 executed Kaggle notebooks (.ipynb/.html/.pdf) and their raw CSVs, in run order;
                               README.md says what each run did and which of its numbers are final
results/
  all_results.csv              final numbers for all 17 datasets (source_run column = which run produced each)
  by_dataset/                  the same rows, one CSV per dataset (02_EMOPain.csv ... 18_SelfRegulationSCP2.csv)
  RESULTS.md                   readable tables: reproduction, improvements, summary, mixer gates
  TIMING.md                    time of every Kaggle run and every step
docs/
  HOW_TO_RUN.md                run the kit on any dataset (Kaggle, Colab or locally)
  DATASETS.md                  the 18 datasets: sources, shapes, splits, ready-to-paste notebook lines
  REPRODUCTION_NOTES.md        reproduction protocol, data sources and the problems we hit
report/
  main.tex, main.pdf           the report (IEEE format), figures/ and the generated timing tables
```

All tables and figures are generated from the raw run outputs, so they can be rebuilt in a few seconds without
re-running any model:

```bash
python src/tables/build_all_results.py
python src/tables/make_results_tables.py
python src/tables/make_timing_table.py
python src/tables/make_figures.py
```

## How to run

```bash
pip install -r requirements.txt
```

**Reproduction (paper protocol).** CPU is enough, except for CAP (an 8.7 GB download and about an hour of CPU):

```bash
python src/reproduce_mantisv2.py --datasets Blink FingerMovements
```

Datasets are downloaded automatically into `data/` (not stored in git).

**Our improvements, on any dataset.** Easiest on Kaggle with the free T4 GPU: upload
[`src/MantisV2_Kit_Kaggle.ipynb`](src/MantisV2_Kit_Kaggle.ipynb), turn on the GPU, edit the "YOUR DATASETS" cell and
run all. Step-by-step guide: [`docs/HOW_TO_RUN.md`](docs/HOW_TO_RUN.md); the source and notebook lines for each of the
18 paper datasets: [`docs/DATASETS.md`](docs/DATASETS.md). Locally:

```bash
python src/mantisv2_kit.py --dataset Blink --source uea                  # all 7 methods
python src/mantisv2_kit.py --dataset Blink --source uea --quick          # 1-3 epochs, to check the setup
```

**The exact runs behind our results.** `src/run_kaggle_parts.py` with `PART = 1 ... 5` is the notebook of runs 05–09;
the scripts in `src/development/` are runs 02–04 (paste each `# %% cell` block into its own Kaggle cell). Each run's
executed notebook is in [`kaggle_runs/`](kaggle_runs/).

## How long it took (Kaggle)

All runs were Kaggle notebooks. The full tables are in [results/TIMING.md](results/TIMING.md), built by
`python src/tables/make_timing_table.py` from the notebooks in `kaggle_runs/` and `results/all_results.csv`. Run 10 (idea 2 for UCI-HAR and Epilepsy-EEG) is not in this table because its notebook was not saved.

| # | `kaggle_runs/` folder | What it did | Hardware | Time |
|---|---|---|---|---|
| 01 | `reproduction_datasets_7-12` | Reproduction of datasets 7–12 (embeddings + Random Forest) | CPU only | 1.99 h |
| 02 | `ideas_1_2_first_try` | First versions of ideas 1 and 2 (4 datasets) | T4 GPU | 35.5 min |
| 03 | `mixer_v2` | Idea 1 (mixer v2) + plain head, datasets 7–12 | T4 GPU | 8.9 min |
| 04 | `idea2_cap_and_lora` | Idea 2 on CAP + idea 3 (LoRA), datasets 7–12 | T4 GPU | 1.17 h |
| 05 | `part1_seven_datasets` | All 7 methods on SCP1, SCP2, EMOPain, MP8, MP50, HHAR-ID/OOD | T4 GPU | 48.0 min |
| 06 | `part2_PCL_failed_out_of_RAM` | PCL-ID/OOD, stopped (out of RAM) | T4 GPU | 24.6 min |
| 07 | `part3_SEDFx-ID` | All 7 methods on SEDFx-ID (211,856 samples) | T4 GPU | 1.99 h |
| 08 | `part4_SEDFx-OOD` | All 7 methods on SEDFx-OOD (186,584 samples) | T4 GPU | 1.82 h |
| 09 | `part5_reruns` | Re-runs after memory fixes (PCL-ID/OOD, LoRA, idea 2 fill-ins) | T4 GPU | 3.30 h |
|   | **Total** | | | **12.2 h** (10.2 h GPU) |

Parts 1–4 ran in parallel, so everything finished in about 14 hours of calendar time (6 Nov 13:48 → 7 Nov 03:56 UTC).

Time per method, summed over the 17 datasets (3 seeds each, features already extracted):

| Step | Total time | Slowest dataset |
|---|---|---|
| Feature extraction (frozen encoder) | 4,242 s (1.2 h) | CAP, 3,386 s on CPU (on the T4 the slowest was PCL-ID, 166 s) |
| RF (paper) | 11,940 s (3.3 h) | SEDFx-ID, 3,753 s |
| LogReg | 1,117 s (19 min) | SEDFx-ID, 378 s |
| Head | 556 s (9 min) | SEDFx-ID, 136 s |
| Mixer v2 (idea 1) | 864 s (14 min) | SEDFx-ID, 240 s |
| Pool (idea 2) | 2,408 s (40 min) | PCL-OOD, 599 s |
| Pool + mixer | 2,813 s (47 min) | PCL-OOD, 727 s |
| LoRA + mixer (idea 3) | 15,322 s (4.3 h) | PCL-OOD, 5,075 s |

The RF times for datasets 7–12 were measured on the CPU-only reproduction notebook, so they are slower than on the
GPU sessions (whose 4 CPU cores run the RF). The mixer head is the cheapest of our improvements: on SEDFx-ID it is
15× faster than the paper's RF and more accurate.

## Data sources

- Blink, FingerMovements: timeseriesclassification.com
- UCI-HAR, Epilepsy-EEG: TF-C preprocessed files on figshare (articles 19930244, 19930199), fetched with the figshare API
- CAP, HHAR, PCL, SEDFx: WOODS preprocessed `.h5` files (Google Drive ids in `docs/DATASETS.md`)
- EMOPain, SelfRegulationSCP1/2: timeseriesclassification.com
- MP8, MP50: Sungu Isiacik et al. (2025), github.com/mlgig/FM4MTSC (public Google Drive folder)
- Ego4D: not used (licence agreement required)

Pretrained model: `paris-noah/MantisV2` from the `mantis-tsfm` package (github.com/vfeofanov/mantis).

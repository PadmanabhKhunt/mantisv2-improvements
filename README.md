# MantisV2: reproduction and improvements

COMP41850 AI for Time Series, group project (UCD, Autumn 2026).

We reproduce the zero-shot results of **MantisV2** (Feofanov et al., 2026, arXiv:2602.17868) and then improve the
model with three changes on top of the pretrained encoder:

1. **Cross-channel mixer.** MantisV2 encodes every channel separately. We add a small, gated attention layer across channels.
2. **Learned token pooling.** The paper uses only layer 2 (CLS + mean token). We learn a weighted mix of all 6 layers
   and attention pooling over the tokens.
3. **LoRA fine-tuning.** Low-rank adapters (rank 8) on the attention weights. This trains 36,864 encoder weights (0.9% of 4.2M).

Status: datasets 7–12 (UCI-HAR, Blink, CAP-ID, CAP-OOD, Epilepsy-EEG, FingerMovements) are done. Datasets 1–6 are
still to run.

## Run it on your datasets (group members start here)

Everything is in **[`kit/`](kit/)**, and the step-by-step guide is **[`kit/HOW_TO_RUN.md`](kit/HOW_TO_RUN.md)**. In short:

1. Download `kit/MantisV2_Kit_Kaggle.ipynb`, import it into Kaggle, and turn on GPU and Internet.
2. Put your dataset names in the "YOUR DATASETS" cell, then **Save & Run All**.
3. Upload the result CSVs to `results/members/<your name>/`. Then `python kit/combine_results.py` builds the 12-dataset table.

The kit runs the paper baseline (`rf`) and all improvements with the same settings as our runs. It accepts UEA/UCR
datasets by name, the TF-C datasets (SleepEEG, Epilepsy, FD-A/B, HAR, Gesture, EMG, ECG), `.ts` files, or your own
numpy arrays.

## Results so far (datasets 7–12, accuracy, mean of 3 seeds)

| Dataset | Paper (MantisV2 + RF) | Our reproduction | Mixer v2 (idea 1) | Learned pooling (idea 2) | Idea 2 + mixer | LoRA + mixer (idea 3) |
|---|---|---|---|---|---|---|
| FingerMovements | 0.550 | 0.550 | 0.543 | 0.533 | — | **0.597** |
| Blink | 0.996 | 0.996 | 0.998 | 0.957 | — | **1.000** |
| UCI-HAR | 0.901 | 0.901 | 0.918 | 0.924 | — | **0.925** |
| Epilepsy-EEG | 0.956 | 0.956 | **0.958** | 0.957 | — | **0.958** |
| CAP-ID | 0.815 | 0.809 | 0.826 | 0.826 | **0.834** | 0.819 |
| CAP-OOD | 0.786 | **0.786** | 0.768 | 0.765 | 0.765 | 0.761 |

"—" means not run. On the four small datasets, idea 2 comes from the first test run (`runs/1_ideas_1_2_first_try`).
LoRA on CAP ran only 4 epochs to fit the free GPU. Full numbers, standard deviations and per-seed values are in
`results/`.

## Repository layout

```
kit/                           ready-to-use version of everything, for any dataset
  mantisv2_kit.py              all methods + data loaders + training (also a command-line tool)
  MantisV2_Kit_Kaggle.ipynb    upload to Kaggle/Colab, edit one cell, run
  HOW_TO_RUN.md                step-by-step guide
  combine_results.py           merges everyone's CSVs into one table
  make_notebook.py             rebuilds the notebook after changes to mantisv2_kit.py
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
results/                       all result CSVs in one place
  datasets_7_12_kit_results.csv   our datasets 7-12 in the kit's format
  members/                     each member's CSVs go here
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
- CAP: WOODS preprocessed `CAP.h5`

Pretrained model: `paris-noah/MantisV2` from the `mantis-tsfm` package (github.com/vfeofanov/mantis).

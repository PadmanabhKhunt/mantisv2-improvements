# How to run MantisV2 and our improvements on any dataset

The kit (`src/mantisv2_kit.py`, or the notebook `src/MantisV2_Kit_Kaggle.ipynb` built from it) runs the paper
baseline and all three improvements on any classification dataset, with the settings we used for all 18 datasets.
You only change the dataset names. For the 18 paper datasets, [`DATASETS.md`](DATASETS.md) lists the `source` to use
for each and ready-to-paste notebook lines. You do not need a GPU on your laptop: the free Kaggle (or
Colab) GPU is enough.

## What it runs

| Method in the results | What it is |
|---|---|
| `rf` | Paper protocol: frozen MantisV2 (layer 2, CLS + mean token) + Random Forest with 200 trees. This is the reproduction. |
| `logreg` | Same features + logistic regression (paper Section 5.3) |
| `head` | Small trained head on the same features, no channel mixing (for the ablation) |
| `mixer` | **Idea 1:** mixer v2, a gated attention layer across channels |
| `pool` | **Idea 2:** learned mix of all 6 encoder layers + attention pooling over tokens |
| `pool_mixer` | **Idea 2 + idea 1** together |
| `lora` | **Idea 3:** LoRA (rank 8) on the encoder's attention weights + mixer v2 head |

Every method runs 3 seeds (0, 1, 2) and reports mean ± std accuracy on the test split.

## Option A: Kaggle (recommended, about 10 minutes of your time)

1. Download the notebook: on GitHub open `src/MantisV2_Kit_Kaggle.ipynb` and click the **Download raw file** button (top right of the file view).
2. On kaggle.com: **Create → New Notebook**, then **File → Import Notebook** and choose the file.
3. In the right-hand panel (**Settings**):
   - **Accelerator → GPU T4 x2**. You need a phone-verified Kaggle account for this.
   - **Internet → On**, because the datasets and the model are downloaded.
4. Edit the cell under **YOUR DATASETS**, for example:
   ```python
   DATASETS = [
       dict(name='SleepEEG', source='tfc', paper=0.xxxx),
       dict(name='BasicMotions', source='uea', paper=0.xxxx),
   ]
   methods = None
   QUICK = False
   ```
   - `paper` is MantisV2's accuracy for that dataset in the paper's tables. It is optional.
   - Tip: first run with `QUICK = True`, which takes a few minutes and checks that the downloads work. Then set it back to `False`.
5. Click **Save Version → Save & Run All (Commit) → Save**. It keeps running if you close the browser.
6. When it finishes, open the notebook's **Output** tab and download `results/all_results.csv`, plus one `<dataset>_results.csv` per dataset.

**Colab instead:** **File → Upload notebook**, then **Runtime → Change runtime type → T4 GPU**, then **Runtime → Run all**.
The CSVs appear in the `results` folder in the file panel on the left.

## Which `source` to use

| Your data | `source` | `name` / `path` |
|---|---|---|
| A dataset on timeseriesclassification.com (UEA multivariate or UCR univariate) | `'uea'` | the exact name on the site, e.g. `'Blink'`, `'ECG200'`, `'BasicMotions'` |
| TF-C preprocessed data (used by the paper for HAR / EEG) | `'tfc'` | `'SleepEEG'`, `'Epilepsy'`, `'FD-A'`, `'FD-B'`, `'HAR'`, `'Gesture'`, `'EMG'`, `'ECG'` |
| A UEA/UCR zip you downloaded yourself (unzipped) | `'ts'` | `name` as above, `path` = the unzipped folder |
| `train.pt` / `test.pt` with `samples` and `labels` (TF-C / UniTS style) | `'pt'` | `path` = the folder |
| A WOODS benchmark `.h5` file (HHAR, PCL, SEDFx, CAP) | `'woods'` | `path` = the `.h5` file, plus `split='ID'` or `split='OOD', train_domains=[...], test_domains=[...]` |
| Anything else | `'numpy'` | `path` = a folder with `X_train.npy`, `y_train.npy`, `X_test.npy`, `y_test.npy` |

For your own data, save numpy arrays like this:
```python
import numpy as np
# X: (samples, channels, time steps); for one channel, (samples, time steps) also works. y: labels (numbers or text).
np.save('X_train.npy', X_train); np.save('y_train.npy', y_train)
np.save('X_test.npy', X_test);   np.save('y_test.npy', y_test)
```
- If there is no official split, save just `X.npy` and `y.npy`. The kit then makes a 75/25 stratified split with seed 0.
- On Kaggle, upload the folder with **Add Input → Upload → New Dataset**. Then use `path='/kaggle/input/<your-dataset-name>'`.

Series of any length are fine: everything is interpolated to 512 steps, as in the paper.

## How long it takes (Kaggle T4)

| Dataset size | Time for all 7 methods |
|---|---|
| Small (under about 5,000 samples, a few channels) | 2–15 minutes |
| Medium (about 10,000 samples) | 20–40 minutes |
| Large like CAP (40,000 samples × 19 channels) | 1.5–2.5 hours; `pool` needs a lot of RAM and is skipped above 20 GB |

To save time, run fewer methods, e.g. `methods = ['rf', 'mixer', 'lora']`.

## Add a new run to the results

1. Put the executed notebook and its `<dataset>_results.csv` files in a new folder `kaggle_runs/NN_<short_name>/`
   (next number). If a dataset and method already exist, the run with the higher number wins.
2. Rebuild everything:
```bash
python src/tables/build_all_results.py     # results/all_results.csv and results/by_dataset/
python src/tables/make_results_tables.py   # results/RESULTS.md
python src/tables/make_timing_table.py     # results/TIMING.md and the report's timing tables
python src/tables/make_figures.py          # report/figures/
```
To get a quick dataset × method table from any folder of result CSVs: `python src/tables/combine_results.py <folder>`.

## Running on your own computer (optional)

```bash
pip install -r requirements.txt
python src/mantisv2_kit.py --dataset Epilepsy --source tfc --paper 0.9558
python src/mantisv2_kit.py --dataset MyData --source numpy --path path/to/folder --methods rf mixer
python src/mantisv2_kit.py --dataset Blink --source uea --quick        # fast check
```
Without a GPU, `rf` and `logreg` are fine, but the trained methods (especially `lora`) are slow.

## If something goes wrong

| Problem | Fix |
|---|---|
| `Torch not compiled with CUDA` or "device cpu" in the log | The GPU is not on: Settings → Accelerator → GPU, then run again |
| Download error / `ConnectionError` | Settings → Internet → On (Kaggle) |
| `HTTPError 401/403` for a `'uea'` dataset | The site sometimes refuses automatic downloads. Download the zip yourself from timeseriesclassification.com, unzip it (on Kaggle: upload it as a dataset), and use `source='ts'` with `path` set to that folder |
| `IndexError` when loading a `'uea'` dataset | The name is wrong; it must match the site exactly (case matters) |
| Out of memory | Use fewer methods (drop `pool`, `pool_mixer`), or ask in the group chat |
| One method shows `FAILED` | The other methods still run; send the log to the group |

If you change `src/mantisv2_kit.py`, run `python src/make_notebook.py` to rebuild the notebook.

"""Builds kit/MantisV2_Kit_Kaggle.ipynb from kit/mantisv2_kit.py, so the notebook always has the latest code.
Run after any change to mantisv2_kit.py:   python kit/make_notebook.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
code = (HERE / 'mantisv2_kit.py').read_text(encoding='utf-8')


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.strip().splitlines(True)}


def py(text):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
            'source': text.strip().splitlines(True)}


cells = [
    md("""
# MantisV2 kit: baseline + our 3 improvements on your datasets

1. **Settings → Accelerator → GPU T4** (Kaggle) or **Runtime → Change runtime type → T4 GPU** (Colab).
2. Edit **only the cell "YOUR DATASETS"** below.
3. Kaggle: **Save Version → Save & Run All (Commit)**. Colab: **Runtime → Run all**.
4. Results: `results/all_results.csv` (Kaggle: Output tab). Send that file to the group.

Cell 2 holds the kit code. Do not edit it; it is generated from `kit/mantisv2_kit.py` in the GitHub repo.
"""),
    py("!pip install -q mantis-tsfm"),
    py("%%writefile mantisv2_kit.py\n" + code),
    md("""
## YOUR DATASETS

One line per dataset. `source` is one of:

- `'uea'`: any dataset on timeseriesclassification.com, by its exact name (e.g. `'Blink'`, `'BasicMotions'`, `'ECG200'`)
- `'tfc'`: TF-C datasets: `'SleepEEG'`, `'Epilepsy'`, `'FD-A'`, `'FD-B'`, `'HAR'`, `'Gesture'`, `'EMG'`, `'ECG'`
- `'ts'`: `.ts` files you downloaded yourself, with `path` = folder containing `<name>_TRAIN.ts` and `<name>_TEST.ts`
- `'numpy'`: your own files, with `path` = folder containing `X_train.npy, y_train.npy, X_test.npy, y_test.npy`
  (or only `X.npy, y.npy`). On Kaggle, upload the folder as a dataset (Add Input) and use its `/kaggle/input/...` path.

`paper` = accuracy of MantisV2 in the paper for that dataset (optional, just stored next to our numbers).
`methods = None` runs everything: rf, logreg, head, mixer, pool, pool_mixer, lora.
`QUICK = True` runs 1-3 epochs to check that everything works (a few minutes); set it back to `False` for real results.
"""),
    py("""
DATASETS = [
    dict(name='BasicMotions', source='uea', paper=None),
    # dict(name='SleepEEG', source='tfc', paper=None),
    # dict(name='MyData', source='numpy', path='/kaggle/input/my-data', paper=None),
]
methods = None          # or e.g. ['rf', 'mixer', 'lora']
QUICK = False
"""),
    py("""
import os, importlib
import pandas as pd
import mantisv2_kit as kit
importlib.reload(kit)

OUT = '/kaggle/working/results' if os.path.exists('/kaggle/working') else 'results'
DATA = '/tmp/data'          # downloads go here, not into the saved output
frames = []
for d in DATASETS:
    try:
        frames.append(kit.run(d['name'], d['source'], d.get('path'), methods, out_dir=OUT, data_dir=DATA,
                              quick=QUICK, paper=d.get('paper')))
    except Exception as e:
        print('FAILED', d['name'], type(e).__name__, e)
if frames:
    df = pd.concat(frames)
    df.to_csv(f'{OUT}/all_results.csv', index=False)
    table = df.pivot_table(index='dataset', columns='method', values='mean').round(4)
    print(table[[m for m in kit.ALL_METHODS if m in table.columns]].to_string())
"""),
]

nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                   'language_info': {'name': 'python'}, 'accelerator': 'GPU'},
      'nbformat': 4, 'nbformat_minor': 5}
out = HERE / 'MantisV2_Kit_Kaggle.ipynb'
out.write_text(json.dumps(nb, indent=1), encoding='utf-8')
print('wrote', out)

"""Collects the final result of every (dataset, method) from the raw CSVs in kaggle_runs/ into one file:

    python src/tables/build_all_results.py
        -> results/all_results.csv               one row per dataset x method, all 17 datasets
        -> results/by_dataset/NN_<dataset>.csv   the same rows, one file per dataset

Every row keeps `source_run`, the kaggle_runs/ folder whose notebook produced it.

Runs 01-04 (datasets 7-12) were made before the kit existed, so their CSVs have other formats; they are converted
here. Runs 05-10 used the kit and already have its format. When a (dataset, method) appears in more than one run, the
later run wins: part 5 re-ran what parts 1-2 could not finish (out of memory), and runs 09/10 replaced the
first-version idea 2 numbers of run 02.
"""
import glob
import os

import pandas as pd

from common import ALL, NUMBER, PAPER, RES, RUNS

COLS = ['no', 'dataset', 'method', 'mean', 'std', 'per_seed', 'paper', 'n_train', 'n_test', 'channels', 'classes',
        'time_s', 'hardware', 'epochs', 'gate', 'layer_weights', 'source_run']
CHANNELS_7_12 = {'UCI-HAR': 3, 'Blink': 4, 'CAP-ID': 19, 'CAP-OOD': 19, 'Epilepsy-EEG': 1, 'FingerMovements': 28}
CLASSES_7_12 = {'UCI-HAR': 6, 'Blink': 2, 'CAP-ID': 6, 'CAP-OOD': 6, 'Epilepsy-EEG': 2, 'FingerMovements': 2}  # paper, Tables 5-6
rows = []


def add(df, run, hardware='T4 GPU'):
    df = df.copy()
    df['source_run'] = run
    df['hardware'] = hardware
    rows.append(df)


# ---- run 01: reproduction of datasets 7-12 (frozen MantisV2 + Random Forest, CPU-only Kaggle session)
r01 = '01_reproduction_datasets_7-12'
rep = pd.read_csv(RUNS / r01 / 'mantisv2_results_datasets_7_12.csv', index_col=0)
add(pd.DataFrame(dict(dataset=rep.index, method='rf', mean=rep.ours_mean.values, std=rep.ours_std.values,
                      per_seed=rep.per_seed.values, n_train=rep.n_train.values, n_test=rep.n_test.values,
                      time_s=rep.rf_s.values)), r01, 'CPU')
info = rep[['n_train', 'n_test']]

# ---- run 02: first try of ideas 1 and 2; only its logistic regression is still a final result
r02 = '02_ideas_1_2_first_try'
d = pd.read_csv(RUNS / r02 / 'improvement_results.csv')
d = d[d.method.str.startswith('LogReg')]
add(pd.DataFrame(dict(dataset=d.dataset, method='logreg', mean=d['mean'], std=d['std'])), r02)

# ---- run 03: plain head ("concat + channel dropout") and mixer v2 (idea 1), datasets 7-12
r03 = '03_mixer_v2'
d = pd.read_csv(RUNS / r03 / 'mixer_v2_results.csv')
d['method'] = d.method.map({'concat + channel dropout': 'head', 'mixer v2': 'mixer'})
add(d[['dataset', 'method', 'mean', 'std', 'per_seed', 'epochs', 'gate', 'time_s']], r03)

# ---- run 04: idea 2 on CAP-ID/OOD and LoRA (idea 3) on datasets 7-12
r04 = '04_idea2_cap_and_lora'
d = pd.read_csv(RUNS / r04 / 'part2_results.csv')
d['method'] = d.method.map(lambda m: 'lora' if 'LoRA' in m else 'pool_mixer' if 'mixer' in m else 'pool')
add(d[['dataset', 'method', 'mean', 'std', 'per_seed', 'epochs', 'gate', 'layer_weights', 'time_s']], r04)

# ---- runs 05-10: the kit (all 7 methods, one CSV per dataset)
for run in sorted(os.listdir(RUNS)):
    if run[:2].isdigit() and int(run[:2]) >= 5:
        for f in sorted(glob.glob(str(RUNS / run / '*_results.csv'))):
            if not os.path.basename(f).startswith('part'):  # part1_all_results.csv repeats the per-dataset files
                add(pd.read_csv(f), run)

df = pd.concat(rows, ignore_index=True).drop_duplicates(['dataset', 'method'], keep='last')
for n, ch in CHANNELS_7_12.items():
    df.loc[df.dataset == n, 'channels'] = ch
    df.loc[df.dataset == n, 'classes'] = CLASSES_7_12[n]
for n in info.index:
    df.loc[df.dataset == n, ['n_train', 'n_test']] = info.loc[n].values
df['paper'] = df.dataset.map(PAPER)
df['no'] = df.dataset.map(NUMBER)
df['method_order'] = df.method.map({m: i for i, m in enumerate(['rf', 'logreg', 'head', 'mixer', 'pool',
                                                                    'pool_mixer', 'lora'])})
df = df.sort_values(['no', 'method_order'])[COLS]
for c in ['no', 'n_train', 'n_test', 'channels', 'classes']:
    df[c] = df[c].astype('Int64')
df.to_csv(ALL, index=False)

out = RES / 'by_dataset'
out.mkdir(exist_ok=True)
for f in out.glob('*.csv'):
    f.unlink()
for (no, name), g in df.groupby(['no', 'dataset']):
    g.to_csv(out / f'{no:02d}_{name}.csv', index=False)
print(f'{len(df)} rows, {df.dataset.nunique()} datasets -> {ALL.relative_to(RES.parent)} and results/by_dataset/')
print(df.groupby('source_run').size().to_string())

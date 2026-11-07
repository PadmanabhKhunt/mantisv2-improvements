"""Shared constants and the loader for results/all_results.csv (used by every script in src/tables/)."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / 'results'
RUNS = ROOT / 'kaggle_runs'
ALL = RES / 'all_results.csv'

# Tables 2 and 3 of the MantisV2 paper, in order: (name, MantisV2 accuracy reported in the paper)
DATASETS = [('Ego4D', 0.5258), ('EMOPain', 0.8798), ('HHAR-ID', 0.9845), ('HHAR-OOD', 0.5822), ('MP8', 0.6857),
            ('MP50', 0.7345), ('UCI-HAR', 0.9013), ('Blink', 0.9956), ('CAP-ID', 0.8152), ('CAP-OOD', 0.7859),
            ('Epilepsy-EEG', 0.9558), ('FingerMovements', 0.5500), ('PCL-ID', 0.5716), ('PCL-OOD', 0.5311),
            ('SEDFx-ID', 0.8000), ('SEDFx-OOD', 0.7636), ('SelfRegulationSCP1', 0.8134), ('SelfRegulationSCP2', 0.5167)]
NUMBER = {n: i for i, (n, _) in enumerate(DATASETS, 1)}
PAPER = dict(DATASETS)
NOT_RUN = {'Ego4D': 'not run: the data needs a signed Ego4D licence agreement, so it is not freely downloadable'}
ORDER = [n for n, _ in DATASETS if n not in NOT_RUN]  # the 17 datasets we ran, in paper order

METHODS = ['rf', 'logreg', 'head', 'mixer', 'pool', 'pool_mixer', 'lora']
NAMES = {'rf': 'RF (paper)', 'logreg': 'LogReg', 'head': 'Head', 'mixer': 'Mixer v2 (idea 1)', 'pool': 'Pooling (idea 2)',
         'pool_mixer': 'Idea 2 + mixer', 'lora': 'LoRA + mixer (idea 3)'}
SHORT = {'SelfRegulationSCP1': 'SCP1', 'SelfRegulationSCP2': 'SCP2', 'FingerMovements': 'FingerMov.',
         'Epilepsy-EEG': 'Epilepsy'}


def load():
    """All final results: one row per (dataset, method), with the Kaggle run it came from."""
    if not ALL.exists():
        raise SystemExit(f'{ALL} is missing: run  python src/tables/build_all_results.py  first')
    return pd.read_csv(ALL)

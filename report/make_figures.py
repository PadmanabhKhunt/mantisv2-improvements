"""Builds the two report figures from the result CSVs in results/.

    python report/make_figures.py      -> report/figures/gain_over_rf.pdf, report/figures/layer_weights.pdf
"""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'report' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

# same file order as kit/make_results_tables.py: later files replace earlier ones
order = lambda f: (0 if 'datasets_7_12' in f else 2 if ('part5' in f or 'idea2' in f) else 1, f)
frames = []
for f in sorted(glob.glob(str(ROOT / 'results' / '**' / '*results.csv'), recursive=True), key=order):
    cols = pd.read_csv(f, nrows=0).columns
    if {'dataset', 'method', 'mean', 'std'} <= set(cols) and 'paper' in cols:
        frames.append(pd.read_csv(f))
df = pd.concat(frames).drop_duplicates(['dataset', 'method'], keep='last')
mean = df.pivot(index='dataset', columns='method', values='mean')

ORDER = ['EMOPain', 'HHAR-ID', 'HHAR-OOD', 'MP8', 'MP50', 'UCI-HAR', 'Blink', 'CAP-ID', 'CAP-OOD', 'Epilepsy-EEG',
         'FingerMovements', 'PCL-ID', 'PCL-OOD', 'SEDFx-ID', 'SEDFx-OOD', 'SelfRegulationSCP1', 'SelfRegulationSCP2']
SHORT = {'SelfRegulationSCP1': 'SCP1', 'SelfRegulationSCP2': 'SCP2', 'FingerMovements': 'FingerMov.',
         'Epilepsy-EEG': 'Epilepsy'}
INK, INK2, GRID, AXIS = '#0b0b0b', '#52514e', '#e1e0d9', '#c3c2b7'
plt.rcParams.update({'font.family': 'serif', 'font.size': 7, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.linewidth': 0.6})

# ---- Figure 1: accuracy gain over the paper's Random Forest, per dataset
series = [('mixer', 'Mixer v2 (idea 1)', '#2a78d6'), ('pool_mixer', 'Idea 2 + mixer', '#eb6834'),
          ('lora', 'LoRA + mixer (idea 3)', '#1baf7a')]
fig, ax = plt.subplots(figsize=(3.45, 4.1))
y = np.arange(len(ORDER))[::-1]
h = 0.26
for k, (m, label, c) in enumerate(series):
    d = 100 * (mean.loc[ORDER, m] - mean.loc[ORDER, 'rf']).values
    ax.barh(y + (1 - k) * h, d, height=h - 0.04, color=c, label=label, linewidth=0)
ax.axvline(0, color=INK2, linewidth=0.7)
ax.set_yticks(y, [SHORT.get(n, n) for n in ORDER])
ax.set_xlabel('Accuracy minus RF (percentage points)')
ax.grid(axis='x', color=GRID, linewidth=0.5)
ax.set_axisbelow(True)
for s in ['top', 'right', 'left']:
    ax.spines[s].set_visible(False)
ax.tick_params(axis='y', length=0)
ax.legend(loc='lower center', bbox_to_anchor=(0.4, 1.0), ncol=3, frameon=False, fontsize=6.3, handlelength=1,
          columnspacing=0.8)
fig.tight_layout(pad=0.3)
fig.savefig(OUT / 'gain_over_rf.pdf')

# ---- Figure 2: learned layer weights of the pooling head (idea 2), mean over 3 seeds
rows = []
for n in ORDER:
    r = df[(df.dataset == n) & (df.method == 'pool')]
    if len(r) and isinstance(r.layer_weights.iloc[0], str):
        rows.append((n, np.mean(json.loads(r.layer_weights.iloc[0]), axis=0)))
W = np.array([w for _, w in rows])
fig, ax = plt.subplots(figsize=(3.45, 2.9))
im = ax.imshow(W, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
    'blue', ['#fcfcfb', '#cde2fb', '#86b6ef', '#2a78d6', '#184f95', '#0d366b']), vmin=0.08, vmax=0.27, aspect='auto')
ax.set_xticks(range(6), [f'L{i}' for i in range(6)])
ax.set_yticks(range(len(rows)), [SHORT.get(n, n) for n, _ in rows])
ax.set_xlabel('Transformer layer (paper uses layer 2 only)')
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
for i in range(W.shape[0]):
    for j in range(6):
        ax.text(j, i, f'{W[i, j]:.2f}', ha='center', va='center', fontsize=5.3,
                color='#ffffff' if W[i, j] > 0.2 else INK)
fig.tight_layout(pad=0.3)
fig.savefig(OUT / 'layer_weights.pdf')
print('saved', sorted(p.name for p in OUT.glob('*.pdf')))
print('datasets without layer weights:', [n for n in ORDER if n not in dict(rows)])

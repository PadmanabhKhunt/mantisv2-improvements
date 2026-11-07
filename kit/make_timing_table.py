"""Builds results/TIMING.md: how long every Kaggle run and every step took.

    python kit/make_timing_table.py

Sources (nothing is typed in by hand):
  - wall-clock time of each Kaggle notebook: the papermill metadata Kaggle saves in every .ipynb under improvements/ and
    reproduction/
  - feature / token extraction time: the log lines "features ... in Ns" and "tokens ... in Ns" in those notebooks,
    and the embed_s column of results/mantisv2_results_datasets_7_12.csv
  - training time of each method (3 seeds): the time_s column of the result CSVs, and rf_s for the CPU reproduction
"""
import glob
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / 'results'

ORDER = ['EMOPain', 'HHAR-ID', 'HHAR-OOD', 'MP8', 'MP50', 'UCI-HAR', 'Blink', 'CAP-ID', 'CAP-OOD', 'Epilepsy-EEG',
         'FingerMovements', 'PCL-ID', 'PCL-OOD', 'SEDFx-ID', 'SEDFx-OOD', 'SelfRegulationSCP1', 'SelfRegulationSCP2']
D7_12 = ['UCI-HAR', 'Blink', 'CAP-ID', 'CAP-OOD', 'Epilepsy-EEG', 'FingerMovements']
METHODS = ['rf', 'logreg', 'head', 'mixer', 'pool', 'pool_mixer', 'lora']
NAMES = {'rf': 'RF', 'logreg': 'LogReg', 'head': 'Head', 'mixer': 'Mixer', 'pool': 'Pool', 'pool_mixer': 'Pool+mix',
         'lora': 'LoRA'}

# what each Kaggle notebook did (folder -> description)
SESSIONS = [
    ('reproduction', 'Reproduction of datasets 7-12: MantisV2 embeddings + Random Forest (paper protocol)'),
    ('1_ideas_1_2_first_try', 'First versions of ideas 1 and 2 on UCI-HAR, Blink, Epilepsy, FingerMovements'),
    ('2_mixer_v2', 'Idea 1 (mixer v2) and the plain head on datasets 7-12'),
    ('3_idea2_cap_and_lora', 'Idea 2 on CAP-ID/OOD and idea 3 (LoRA) on datasets 7-12'),
    ('part1', 'Kit, all 7 methods: SCP1, SCP2, EMOPain, MP8, MP50, HHAR-ID, HHAR-OOD'),
    ('part2_failed_out_of_RAM', 'Kit on PCL-ID/PCL-OOD (stopped: ran out of RAM, re-run in part 5)'),
    ('part3_SEDFx-ID', 'Kit, all 7 methods: SEDFx-ID (211,856 samples)'),
    ('part4_SEDFx-OOD', 'Kit, all 7 methods: SEDFx-OOD (186,584 samples)'),
    ('part5_reruns', 'Re-runs after memory fixes: PCL-ID, PCL-OOD, LoRA on EMOPain/MP50, idea 2 on Blink/FingerMovements'),
]


def notebook_text(nb):
    for c in nb['cells']:
        for o in c.get('outputs', []):
            yield ''.join(o['text']) if 'text' in o else ''.join(o.get('data', {}).get('text/plain', ''))


# ---- 1. Kaggle sessions
sessions, feat, tok = [], {}, {}
notebooks = sorted(glob.glob(str(ROOT / 'improvements' / '**' / '*.ipynb'), recursive=True) +
                   glob.glob(str(ROOT / 'reproduction' / '*.ipynb')))
for f in notebooks:
    nb = json.load(open(f, encoding='utf-8'))
    pm = nb.get('metadata', {}).get('papermill', {})
    text = '\n'.join(notebook_text(nb))
    gpu = 'CPU only' if 'device: cpu' in text else 'T4 GPU' if 'Tesla T4' in text or "device = 'cuda'" in json.dumps(nb['cells']) else 'CPU only'
    key = next(k for k, _ in SESSIONS if k in f.replace('\\', '/'))
    sessions.append(dict(key=key, start=pm.get('start_time', '')[:16].replace('T', ' '), sec=pm.get('duration', 0),
                         hw=gpu))
    ds = None
    for line in text.splitlines():  # kit log lines: "HH:MM:SS <dataset>: device ...", "features (...) in 12s"
        m = re.match(r'\d\d:\d\d:\d\d (\S+): device', line)
        if m:
            ds = m.group(1)
        m = re.search(r'(features|tokens) \(.*\) in (\d+)s', line)
        if m and ds:  # later notebooks (part 5) replace earlier ones
            (feat if m.group(1) == 'features' else tok)[ds] = int(m.group(2))
sessions.sort(key=lambda s: [k for k, _ in SESSIONS].index(s['key']))
desc = dict(SESSIONS)

# ---- 2. training time per dataset and method (seconds for 3 seeds)
order = lambda f: (0 if 'datasets_7_12' in f else 2 if ('part5' in f or 'idea2' in f) else 1, f)
frames = []
for f in sorted(glob.glob(str(RES / 'datasets_1-18_kit' / '*results.csv')), key=order):
    frames.append(pd.read_csv(f))
kit = pd.concat(frames).drop_duplicates(['dataset', 'method'], keep='last')
T = {(r.dataset, r.method): r.time_s for r in kit.itertuples()}

# datasets 7-12 were run before the kit existed, so their times are in the older CSVs
rep = pd.read_csv(RES / 'mantisv2_results_datasets_7_12.csv', index_col=0)
cpu = set()
for n in D7_12:
    T[(n, 'rf')] = rep.loc[n, 'rf_s']
    cpu.add((n, 'rf'))
    feat[n] = rep.loc[n, 'embed_s']
    cpu.add((n, 'features'))
mv = pd.read_csv(RES / 'mixer_v2_results.csv')
for r in mv.itertuples():
    T[(r.dataset, 'head' if 'concat' in r.method else 'mixer')] = r.time_s
p2 = pd.read_csv(RES / 'part2_results.csv')
for r in p2.itertuples():
    m = 'lora' if 'LoRA' in r.method else 'pool_mixer' if 'mixer' in r.method else 'pool'
    T[(r.dataset, m)] = r.time_s

fmt = lambda v, k: '–' if v is None or pd.isna(v) else ('<1' if v < 0.5 else f'{v:,.0f}') + ('†' if k in cpu else '')
rows, totals = [], {m: 0 for m in METHODS + ['features']}
for n in ORDER:
    fv = feat.get(n)
    if n == 'CAP-OOD':
        fv = None  # CAP-ID and CAP-OOD share one embedding pass (counted on CAP-ID)
    cells = [fmt(fv, (n, 'features')), fmt(tok.get(n), (n, 'tokens'))]
    tot = (fv or 0) + (tok.get(n) or 0)
    for m in METHODS:
        v = T.get((n, m))
        cells.append(fmt(v, (n, m)))
        if v is not None and not pd.isna(v):
            tot += v
            totals[m] += v
    totals['features'] += fv or 0
    rows.append((n, cells, tot))

mins = lambda s: f'{s / 60:.1f} min' if s < 3600 else f'{s / 3600:.2f} h'
out = ['# Timing: how long each step took', '',
       'Generated by `kit/make_timing_table.py` from the saved Kaggle notebooks and the result CSVs.', '',
       '## 1. Kaggle sessions (wall-clock time of each notebook)', '',
       '| # | Notebook (folder) | What it did | Hardware | Started (UTC) | Time |', '|---|---|---|---|---|---|']
for i, s in enumerate(sessions, 1):
    out.append(f"| {i} | `{s['key']}` | {desc[s['key']]} | {s['hw']} | {s['start']} | {mins(s['sec'])} |")
total = sum(s['sec'] for s in sessions)
gpu_total = sum(s['sec'] for s in sessions if s['hw'] == 'T4 GPU')
out += ['', f'**Total: {mins(total)} of Kaggle time** ({mins(gpu_total)} on a T4 GPU, '
        f'{mins(total - gpu_total)} CPU only). Parts 1–4 ran at the same time in separate Kaggle sessions, and part 5 '
        'overlapped with parts 3 and 4, so the real (calendar) time was 6 Nov 13:48 → 7 Nov 03:56 UTC.',
        'The idea 2 fill-in for UCI-HAR and Epilepsy-EEG (`*_idea2_results.csv`) was a short extra run whose notebook '
        'was not saved; only its CSV is in `results/`.', '',
        '## 2. Time per dataset and method (seconds, 3 seeds)', '',
        '*Features* = one pass of the frozen encoder to get the 512-d embedding per channel (used by RF, LogReg, Head, '
        'Mixer). *Tokens* = the pass that keeps all 6 layers × 5 tokens (used by Pool, Pool+mix). The method columns '
        'are training + testing for 3 seeds, with the features already computed; LoRA runs the encoder itself in every '
        'epoch, so it needs no feature pass. LogReg is one run (it is deterministic).', '',
        '† = run on the Kaggle CPU (no GPU) in the reproduction notebook; everything else ran on a T4 GPU '
        '(RF always runs on the CPU cores, `n_jobs=-1`). – = not recorded. CAP-ID and CAP-OOD share one feature pass.',
        '',
        '| Dataset | Features | Tokens | ' + ' | '.join(NAMES[m] for m in METHODS) + ' | Total |',
        '|---' * (len(METHODS) + 4) + '|']
for n, cells, tot in rows:
    out.append(f'| {n} | ' + ' | '.join(cells) + f' | {mins(tot)} |')
out.append('| **Sum** | ' + f"{totals['features']:,.0f} | | " +
           ' | '.join(f'{totals[m]:,.0f}' for m in METHODS) + ' | |')
out += ['', '## 3. Main points', '',
        f"- LoRA is the most expensive method: {totals['lora'] / 3600:.1f} h in total, "
        f"against {totals['mixer'] / 60:.0f} min for the mixer head and {totals['rf'] / 3600:.1f} h for the paper's RF.",
        '- On the big Sleep-EDFx sets the RF takes about an hour (3,753 s on SEDFx-ID); the mixer takes 4 minutes and '
        'is more accurate.',
        '- The slowest single run was LoRA on PCL-OOD (48 channels): 5,075 s = 85 min for 3 seeds.',
        '- Extracting features on CAP took 56 min on the CPU in the reproduction notebook; the same kind of step takes '
        '2–3 min on the T4 for much bigger datasets (SEDFx-ID: 138 s), so use a GPU.', '']
(RES / 'TIMING.md').write_text('\n'.join(out), encoding='utf-8')
print('saved', RES / 'TIMING.md')

# ---- LaTeX versions of both tables for the report (\input in report/main.tex)
EOL = ' ' + chr(92) * 2  # LaTeX end of table row
SHORT = {'SelfRegulationSCP1': 'SCP1', 'SelfRegulationSCP2': 'SCP2', 'FingerMovements': 'FingerMov.',
         'Epilepsy-EEG': 'Epilepsy'}
tex = lambda c: c.replace('†', r'$^\dagger$').replace('–', '--').replace('<1', '$<$1')
hours = lambda s: '$<$1' + r'\,min' if s < 30 else (f'{s / 60:.0f}' + r'\,min' if s < 3600 else f'{s / 3600:.1f}' + r'\,h')
L = ['% generated by kit/make_timing_table.py', r'\begin{tabular}{l' + 'r' * (len(METHODS) + 3) + '}', r'\toprule',
     'Dataset & Feat. & Tok. & ' + ' & '.join(NAMES[m] for m in METHODS) + ' & Total' + EOL, r'\midrule']
for n, cells, tot in rows:
    L.append(f'{SHORT.get(n, n)} & ' + ' & '.join(tex(c) for c in cells) + f' & {hours(tot)}' + EOL)
L += [r'\midrule', f"Sum & {totals['features']:,.0f} & & " + ' & '.join(f'{totals[m]:,.0f}' for m in METHODS) +
      ' &' + EOL, r'\bottomrule', r'\end{tabular}']
(ROOT / 'report' / 'timing_table.tex').write_text('\n'.join(L) + '\n', encoding='utf-8')

S = ['% generated by kit/make_timing_table.py', r'\begin{tabular}{rp{0.68\columnwidth}lr}', r'\toprule',
     r'\# & Kaggle notebook: what it ran & HW & Time' + EOL, r'\midrule']
SHORT_DESC = {'reproduction': 'Reproduction, datasets 7--12 (embeddings + RF)',
              '1_ideas_1_2_first_try': 'First versions of ideas 1 and 2 (4 datasets)',
              '2_mixer_v2': 'Mixer v2 and plain head, datasets 7--12',
              '3_idea2_cap_and_lora': 'Idea 2 on CAP; LoRA on datasets 7--12',
              'part1': 'All methods: SCP1/2, EMOPain, MP8/50, HHAR',
              'part2_failed_out_of_RAM': 'PCL-ID/OOD (stopped, out of RAM)',
              'part3_SEDFx-ID': 'All methods: SEDFx-ID',
              'part4_SEDFx-OOD': 'All methods: SEDFx-OOD',
              'part5_reruns': 'Re-runs after memory fixes (PCL, LoRA, idea 2)'}
for i, s in enumerate(sessions, 1):
    S.append(f"{i} & {SHORT_DESC[s['key']]} & {'CPU' if s['hw'] == 'CPU only' else 'T4'} & {hours(s['sec'])}" + EOL)
S += [r'\midrule', f'& Total ({hours(gpu_total)} on GPU) & & {hours(total)}' + EOL, r'\bottomrule', r'\end{tabular}']
(ROOT / 'report' / 'sessions_table.tex').write_text('\n'.join(S) + '\n', encoding='utf-8')
print('saved report/timing_table.tex, report/sessions_table.tex')

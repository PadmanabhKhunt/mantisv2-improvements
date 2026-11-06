# Kaggle runner for datasets 1-6 and 13-18 (cell 2 of the notebook; cell 1 is "%%writefile mantisv2_kit.py" + the kit).
# The same notebook is run 4 times with PART = 1, 2, 3, 4 so the parts run in parallel on Kaggle:
#   PART 1: SelfRegulationSCP1, SelfRegulationSCP2, EMOPain, MP8, MP50, HHAR-ID, HHAR-OOD
#   PART 2: PCL-ID, PCL-OOD        PART 3: SEDFx-ID        PART 4: SEDFx-OOD
#   PART 5: re-runs after two memory fixes (LoRA batch in fit(), token array in tokens()): LoRA for EMOPain and MP50,
#           pool/pool_mixer/LoRA for PCL-ID, everything for PCL-OOD (part 2 ran out of RAM), idea 2 for Blink/FingerMovements
# Ego4D (dataset 1) is not run: it needs a signed licence agreement.
PART = 1
!pip install -q mantis-tsfm gdown
import os, zipfile, subprocess, glob, numpy as np, pandas as pd, gdown
import mantisv2_kit as kit
DATA, OUT = '/tmp/data', '/kaggle/working/results'
os.makedirs(DATA, exist_ok=True)

def get(fn):
    try:
        fn()
    except Exception as e:
        print('DATA FAILED:', type(e).__name__, e, flush=True)

def uea(n):
    subprocess.run(['wget', '-q', '-O', f'{DATA}/{n}.zip', f'https://www.timeseriesclassification.com/aeon-toolkit/{n}.zip'], check=True)
    zipfile.ZipFile(f'{DATA}/{n}.zip').extractall(f'{DATA}/{n}')

def mp():
    # MP8 / MP50 as released by Sungu Isiacik et al. (2025), github.com/mlgig/FM4MTSC
    gdown.download_folder(id='10Hvep0hk9naj_nRk4JMiXHMeV1vw0lcd', output=f'{DATA}/fm4mtsc', quiet=True)
    for n in ['MP8', 'MP50']:
        d = np.load(f'{DATA}/fm4mtsc/{n}.npy', allow_pickle=True).item()
        print(n, list(d), {k: np.shape(v['X']) for k, v in d.items()}, flush=True)
        os.makedirs(f'{DATA}/{n}', exist_ok=True)
        for s in ['train', 'test']:
            np.save(f'{DATA}/{n}/X_{s}.npy', np.asarray(d[s]['X'], dtype=np.float32))
            np.save(f'{DATA}/{n}/y_{s}.npy', np.asarray(d[s]['y']))

def woods(gid, name):
    # WOODS benchmark files (github.com/jc-audet/WOODS, woods/download.py)
    gdown.download(id=gid, output=f'{DATA}/{name}.zip', quiet=True)
    zipfile.ZipFile(f'{DATA}/{name}.zip').extractall(f'{DATA}/{name}')
    os.remove(f'{DATA}/{name}.zip')

def h5(name):
    return [os.path.join(r, f) for r, _, fs in os.walk(f'{DATA}/{name}') for f in fs if f.endswith('.h5')][0]

if PART == 1:
    for n in ['SelfRegulationSCP1', 'SelfRegulationSCP2', 'EMOPain']:
        get(lambda: uea(n))
    get(mp)
    get(lambda: woods('1Z3IcrCE-o77p4YrvkCy-Y-0CgCyxVHet', 'HHAR'))
    JOBS = [dict(name='SelfRegulationSCP1', source='uea', paper=0.8134),
            dict(name='SelfRegulationSCP2', source='uea', paper=0.5167),
            dict(name='EMOPain', source='uea', paper=0.8798),
            dict(name='MP8', source='numpy', paper=0.6857),
            dict(name='MP50', source='numpy', paper=0.7345),
            dict(name='HHAR-ID', source='woods', h5='HHAR', split='ID', paper=0.9845),
            dict(name='HHAR-OOD', source='woods', h5='HHAR', split='OOD', train_domains=['nexus4', 's3', 's3mini'],
                 test_domains=['lgwatch'], paper=0.5822)]
elif PART == 5:
    # re-runs after the memory fixes (part 1: LoRA out of GPU memory; part 2: tokens out of RAM)
    for n in ['EMOPain', 'Blink', 'FingerMovements']:
        get(lambda: uea(n))
    get(mp)
    get(lambda: woods('118DNxHpzeJwVTM22wzZhSiOuDsno0nay', 'PCL'))
    JOBS = [dict(name='EMOPain', source='uea', methods=['lora'], paper=0.8798),
            dict(name='MP50', source='numpy', methods=['lora'], paper=0.7345),
            dict(name='PCL-ID', source='woods', h5='PCL', split='ID', methods=['pool', 'pool_mixer', 'lora'], paper=0.5716),
            dict(name='PCL-OOD', source='woods', h5='PCL', split='OOD', train_domains=['Cho2017'],
                 test_domains=['Lee2019_MI'], paper=0.5311),
            # idea 2 with the current head for two of our datasets 7-12 (the site refuses downloads from our laptop)
            dict(name='Blink', source='uea', methods=['pool', 'pool_mixer'], paper=0.9956),
            dict(name='FingerMovements', source='uea', methods=['pool', 'pool_mixer'], paper=0.5500)]
elif PART == 2:
    get(lambda: woods('118DNxHpzeJwVTM22wzZhSiOuDsno0nay', 'PCL'))
    JOBS = [dict(name='PCL-ID', source='woods', h5='PCL', split='ID', paper=0.5716),
            dict(name='PCL-OOD', source='woods', h5='PCL', split='OOD', train_domains=['Cho2017'],
                 test_domains=['Lee2019_MI'], paper=0.5311)]
else:
    get(lambda: woods('15j_bsiOmMJb42mG712Vhv3jZ4MQSOgoT', 'SEDFx'))
    JOBS = [dict(name='SEDFx-ID', source='woods', h5='SEDFx', split='ID', paper=0.8000)] if PART == 3 else \
           [dict(name='SEDFx-OOD', source='woods', h5='SEDFx', split='OOD', train_domains=['Age 20-40', 'Age 40-60'],
                 test_domains=['Age 80-100'], paper=0.7636)]

print(sorted(os.listdir(DATA)), flush=True)
for j in JOBS:
    j = dict(j)
    try:
        if 'h5' in j:
            j['path'] = h5(j.pop('h5'))
        kit.run(j.pop('name'), j.pop('source'), j.pop('path', None), out_dir=OUT, data_dir=DATA, **j)
    except Exception as e:
        print('RUN FAILED:', type(e).__name__, e, flush=True)
files = glob.glob(f'{OUT}/*_results.csv')
if files:
    df = pd.concat([pd.read_csv(f) for f in files])
    df.to_csv(f'{OUT}/part{PART}_all_results.csv', index=False)
    print(df.pivot_table(index='dataset', columns='method', values='mean').round(4).to_string())

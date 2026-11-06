# The 18 datasets and who runs them

The 18 datasets are the rows of the MantisV2 paper's Table 2 (HAR) and Table 3 (EEG), in order. Shapes come from
Tables 5 and 6 (Appendix A.1). `paper` is the MantisV2 column of Tables 2 and 3: put it in the notebook so it shows next
to our numbers.

| # | Dataset | Who | Channels × length | Train / test | Classes | Paper (MantisV2) | Kit `source` |
|---|---|---|---|---|---|---|---|
| 1 | Ego4D | Aadil | 6 × 1000 | 247,095 / 66,994 | 31 | 0.5258 | `numpy` |
| 2 | EMOPain | Aadil | 30 × 200 | 968 / 355 | 3 | 0.8798 | `numpy` or `pt` |
| 3 | HHAR-ID | Aadil | 6 × 500 | 8,716 / 3,419 | 6 | 0.9845 | `woods`, `split='ID'` |
| 4 | HHAR-OOD | Aadil | 6 × 500 | 11,150 / 2,154 | 6 | 0.5822 | `woods`, `split='OOD'` |
| 5 | MP8 | Aadil | 8 × 161 | 1,426 / 595 | 4 | 0.6857 | `numpy` |
| 6 | MP50 | Aadil | 50 × 161 | 1,426 / 595 | 4 | 0.7345 | `numpy` |
| 7 | UCI-HAR | Padmanabh ✅ | 3 × 206 | 5,881 / 2,947 | 6 | 0.9013 | `tfc` (`'HAR'`) |
| 8 | Blink | Padmanabh ✅ | 4 × 510 | 500 / 450 | 2 | 0.9956 | `uea` |
| 9 | CAP-ID | Padmanabh ✅ | 19 × 3000 | 25,748 / 10,098 | 6 | 0.8152 | `woods`, `split='ID'` |
| 10 | CAP-OOD | Padmanabh ✅ | 19 × 3000 | 27,393 / 8,265 | 6 | 0.7859 | `woods`, `split='OOD'` |
| 11 | Epilepsy-EEG | Padmanabh ✅ | 1 × 178 | 60 / 11,420 | 2 | 0.9558 | `tfc` (`'Epilepsy'`) |
| 12 | FingerMovements | Padmanabh ✅ | 28 × 50 | 316 / 100 | 2 | 0.5500 | `uea` |
| 13 | PCL-ID | Praneet | 48 × 750 | 14,405 / 5,650 | 2 | 0.5716 | `woods`, `split='ID'` |
| 14 | PCL-OOD | Praneet | 48 × 750 | 9,880 / 7,800 | 2 | 0.5311 | `woods`, `split='OOD'` |
| 15 | SEDFx-ID | Praneet | 4 × 3000 | 152,178 / 59,678 | 6 | 0.8000 | `woods`, `split='ID'` |
| 16 | SEDFx-OOD | Praneet | 4 × 3000 | 133,746 / 52,838 | 6 | 0.7636 | `woods`, `split='OOD'` |
| 17 | SelfRegulationSCP1 | Praneet | 6 × 896 | 268 / 293 | 2 | 0.8134 | `uea` |
| 18 | SelfRegulationSCP2 | Praneet | 7 × 1152 | 200 / 180 | 2 | 0.5167 | `uea` |

**Check after loading:** the kit's log prints `train N, test N, channels C, classes K` for each dataset. These should
match the table above. If they do not, the data or the split is different from the paper; write that down, because it
is useful for the report.

---

## The same 5 steps for every dataset

1. **Get the data** from the source the paper names (Appendix A.1, also listed per dataset below).
2. **Put it in a form the kit reads** (the `source` column above):
   - `uea`: nothing to do, the kit downloads it by name.
   - `woods`: the WOODS `.h5` file as it is.
   - `numpy`: four files, `X_train.npy`, `y_train.npy`, `X_test.npy`, `y_test.npy`, with X shaped (samples, channels, time).
   - `pt`: `train.pt` and `test.pt`, each a dict with `samples` and `labels`.
3. **Make it reachable from Kaggle:**
   - Either upload it with **Add Input → Upload → New Dataset**. The path is then `/kaggle/input/<dataset-name>/...`
   - Or download it inside the notebook into `/tmp/data/` in an extra cell before the "YOUR DATASETS" cell (better for big files).
4. **Write one line per dataset** in the "YOUR DATASETS" cell (ready-made lines are below), then **Save & Run All**.
5. **Upload the result CSVs** to `results/members/<your name>/` on GitHub (see `HOW_TO_RUN.md`).

Do each new dataset first with `QUICK = True` (a few minutes). This catches path and format mistakes before a long run.

### Converting your data to numpy (any format)

Load the data however its source provides it, then save four arrays:
```python
import numpy as np
# X: float array, shape (samples, channels, time). If yours is (samples, time, channels), use X.transpose(0, 2, 1).
# y: labels, numbers or strings.
np.save('X_train.npy', X_train.astype('float32')); np.save('y_train.npy', y_train)
np.save('X_test.npy',  X_test.astype('float32'));  np.save('y_test.npy',  y_test)
```
- Use the **official train/test split** if the dataset has one.
- If it has a validation split too, leave it out, as the paper does.
- For a very large dataset like Ego4D, `X.astype('float16')` halves the memory; the kit accepts it.

### WOODS datasets (HHAR, PCL, SEDFx)

These come from the WOODS benchmark (github.com/jc-audet/WOODS). Its `woods/download.py` lists where each `.h5` file
is. We used the same approach for CAP: `improvements/code/3_idea2_cap_and_lora.py`, cell 4, downloads the file with
`gdown` into `/tmp/data` and unzips it.

- **ID split:** the kit merges all domains, shuffles them with seed 0, and takes the first 63.75% as train and the last 25% as test. This is exactly what we did for CAP-ID; the paper gives no seed.
- **OOD split:** you name the training and test domains. The domain names inside the file may be written differently from the paper. On the first run the log prints them (`... domains: [...]`); copy the names from there. A wrong name stops with an error that lists the correct ones.

---

## Aadil: datasets 1–6

```python
DATASETS = [
    dict(name='Ego4D',    source='numpy', path='/kaggle/input/<ego4d-folder>', paper=0.5258),
    dict(name='EMOPain',  source='numpy', path='/kaggle/input/<emopain-folder>', paper=0.8798),
    dict(name='HHAR-ID',  source='woods', path='/tmp/data/<HHAR file>.h5', split='ID', paper=0.9845),
    dict(name='HHAR-OOD', source='woods', path='/tmp/data/<HHAR file>.h5', split='OOD',
         train_domains=['nexus4', 's3', 's3mini'], test_domains=['lgwatch'], paper=0.5822),
    dict(name='MP8',      source='numpy', path='/kaggle/input/<mp8-folder>', paper=0.6857),
    dict(name='MP50',     source='numpy', path='/kaggle/input/<mp50-folder>', paper=0.7345),
]
```

- **Ego4D:** using it needs the Ego4D licence agreement, so sign that first. The paper takes the IMU signals and preprocesses them with the script from Chen et al. (2025). Save the result as numpy.
  - It is very large (314,000 samples), so give it its own notebook and run it in parts, one Kaggle version each (each Kaggle run stops after 12 hours):
    - version 1: `methods = ['rf', 'logreg']` (the Random Forest alone may take several hours on 4 CPU cores);
    - version 2: `methods = ['head', 'mixer']`;
    - version 3: `methods = ['lora']`.
  - `pool` is skipped automatically, because its tokens would need about 28 GB of RAM.
- **EMOPain:** the paper downloads it with the script from Gao et al. (2024), the UniTS repository. If that gives `.pt` files with `samples`/`labels`, use `source='pt'`; otherwise convert to numpy.
- **HHAR-ID / HHAR-OOD:** one WOODS file gives both. The paper's OOD split is train on "nexus4", "s3" and "s3mini", test on "lgwatch". Check the exact names in the log.
- **MP8 / MP50:** Military Press, in the versions by Sungu Isiacik et al. (2025). MP8 has 8 key body points and MP50 has all 50 coordinates. Same samples, so they share the split.

## Praneet: datasets 13–18

```python
DATASETS = [
    dict(name='SelfRegulationSCP1', source='uea', paper=0.8134),
    dict(name='SelfRegulationSCP2', source='uea', paper=0.5167),
    dict(name='PCL-ID',    source='woods', path='/tmp/data/<PCL file>.h5', split='ID', paper=0.5716),
    dict(name='PCL-OOD',   source='woods', path='/tmp/data/<PCL file>.h5', split='OOD',
         train_domains=['<Cho2017>'], test_domains=['<Lee2019MI>'], paper=0.5311),
    dict(name='SEDFx-ID',  source='woods', path='/tmp/data/<SEDFx file>.h5', split='ID', paper=0.8000),
    dict(name='SEDFx-OOD', source='woods', path='/tmp/data/<SEDFx file>.h5', split='OOD',
         train_domains=['<age groups 20-60>'], test_domains=['<age group 80-100>'], paper=0.7636),
]
```

- **SelfRegulationSCP1 / SCP2:** easiest, the kit downloads them. Start with these two to learn the workflow (about 10 minutes).
- **PCL:** the paper's OOD split is train on "Cho2017", validation on "PhysionetMI" (not used), test on "Lee2019MI".
- **SEDFx:** the paper's OOD split is train on ages 20–60, validation on 60–80 (not used), test on 80–100. If the file stores smaller age groups, list every group that falls in 20–60 in `train_domains`.
- **SEDFx is the biggest dataset** (212,000 samples × 4 × 3000). Run SEDFx-ID and SEDFx-OOD in separate notebook versions. It is about 5 times CAP, which took 70 minutes on a T4, so expect several hours each. Split methods across versions as described for Ego4D if a run gets close to 12 hours.

---

## Time and GPU budget (Kaggle gives 30 GPU hours per person per week)

| Dataset | Rough time for all methods on a T4 |
|---|---|
| SCP1, SCP2, EMOPain, MP8, MP50 | 5–20 minutes each |
| HHAR-ID/OOD, PCL-ID/OOD | 30–90 minutes each |
| SEDFx-ID/OOD, Ego4D | several hours each, so split by methods |

If you are short on time, the methods that matter most for the report are `rf` (the reproduction), `mixer` (idea 1)
and `lora` (idea 3).

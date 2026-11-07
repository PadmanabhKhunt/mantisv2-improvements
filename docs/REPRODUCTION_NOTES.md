# MantisV2 reproduction — datasets 7–12 (Padmanabh's share)

Run on 6 Nov 2026, Google Colab (Tesla T4 GPU, Python 3.13), notebook
`MantisV2_Reproduction_Datasets_7-12_Padmanabh.ipynb` in Padmanabh's Google Drive
(https://colab.research.google.com/drive/16mi0PGWg-hHSUL-BoEAVT5veu6RLIiDv).

Paper: Feofanov et al. (2026), *MantisV2: Closing the Zero-Shot Gap in Time Series Classification with
Synthetic Data and Test-Time Strategies*, arXiv:2602.17868 — Tables 2 (HAR) and 3 (EEG), MantisV2 column.
Datasets 7–12 = Table 2 + Table 3 read in order: UCI-HAR, Blink, CAP-ID, CAP-OOD, Epilepsy-EEG, FingerMovements.

**Update (same day):** the Colab compute quota ran out while CAP was running, so the full run moved to Kaggle:
private notebook *MantisV2 Reproduction Datasets 7-12 Padmanabh*
(https://www.kaggle.com/code/padmanabhkhunt/mantisv2-reproduction-datasets-7-12-padmanabh), CPU session
(4 cores, 31 GB RAM), run as "Save & Run All (Commit)" version 1. In an interactive test on Kaggle, FingerMovements
(0.5500 ± 0.0082), Blink (0.9956 ± 0.0018) and UCI-HAR seed 0 (0.9033) gave **identical** numbers to Colab —
the pipeline is deterministic across machines. CAP embeddings are saved as `cap_embeddings_Machine*.npz`
in the Kaggle output for the improvement experiments. A standalone script version is at
`06 - Code and Notebooks/MantisV2 Reproduction/reproduce_mantisv2.py` (for the GitHub repo).

## Protocol (from paper + official repo README)

- Code: `pip install mantis-tsfm` (github.com/vfeofanov/mantis), checkpoint `paris-noah/MantisV2` (4.19M params, paper says 4.2M).
- `MantisV2(return_transf_layer=2, output_token='combined')` → 512-dim embedding per channel (cls token ‖ mean token).
  The repo tutorial re-sets both attributes after `from_pretrained`; we did the same.
- Every series linearly interpolated to length 512; each channel encoded independently, embeddings concatenated.
- Random Forest, 200 trees, unlimited depth, `random_state` = 0, 1, 2 → mean ± std (paper: "3 seeds").

## Data sources

| # | Dataset | Source used | Shape check vs paper Tables 5/6 |
|---|---|---|---|
| 7 | UCI-HAR | TF-C preprocessed (figshare article 19930244, `train.pt`/`test.pt`, via figshare API) | train (5881, 3, 206), test 2947 — matches |
| 8 | Blink | timeseriesclassification.com `Blink.zip` (same as UniTS script) | train (500, 4, 510), test 450 — matches |
| 9 | CAP-ID | WOODS preprocessed `CAP.h5` (Google Drive id 1NFwX2CqLrenWF4az0c6J-OglAoD48PAT, 8.66 GB zip / 9.76 GB h5) | 40,390 samples total; train 25,748 / test 10,098 — matches |
| 10 | CAP-OOD | same file: Machines 0,1,3 train / Machine 2 val (unused) / Machine 4 test | train 27,393, test 8,265 — matches |
| 11 | Epilepsy-EEG | TF-C preprocessed (figshare article 19930199) | train (60, 1, 178), test 11,420 — matches |
| 12 | FingerMovements | timeseriesclassification.com `FingerMovements.zip` (UEA) | train (316, 28, 50), test 100 — matches |

## Results (accuracy, 3 RF seeds) — final Kaggle run (version 1, 7,186 s, CPU only)

| Dataset | Ours | Paper (MantisV2) | Diff | Per seed (0 / 1 / 2) |
|---|---|---|---|---|
| UCI-HAR | 0.9013 ± 0.0015 | 0.9013 ± 0.002 | 0.0000 | 0.9033 / 0.8999 / 0.9006 |
| Blink | 0.9956 ± 0.0018 | 0.9956 ± 0.002 | 0.0000 | 0.9978 / 0.9956 / 0.9933 |
| CAP-ID | 0.8091 ± 0.0010 | 0.8152 ± 0.001 | **−0.0061** | 0.8106 / 0.8086 / 0.8082 |
| CAP-OOD | 0.7858 ± 0.0018 | 0.7859 ± 0.000 | −0.0001 | 0.7851 / 0.7840 / 0.7883 |
| Epilepsy-EEG | 0.9558 ± 0.0012 | 0.9558 ± 0.001 | 0.0000 | 0.9542 / 0.9561 / 0.9570 |
| FingerMovements | 0.5500 ± 0.0082 | 0.5500 ± 0.010 | 0.0000 | 0.54 / 0.56 / 0.55 |

5 of 6 match the paper to 3–4 decimals. CAP-ID is 0.6 points lower — the only dataset whose split depends on an
unreported random seed (issue 3 below), so a different train/test draw is the most likely cause; CAP-OOD uses the same
embeddings with a fixed machine split and matches, which suggests the embeddings themselves are correct.
(The earlier Colab run gave UCI-HAR 0.9014 ± 0.0014; Kaggle 0.9013 — tiny CPU/GPU float difference.)

Timing on Kaggle CPU (torch used only 2 threads): CAP embedding 3,386 s for 40,390 samples; RF 1,596 s (CAP-ID) and
1,681 s (CAP-OOD) for 3 seeds.

Local copies (in `06 - Code and Notebooks/MantisV2 Reproduction/`): executed notebook `.ipynb`, `.html`, `.pdf`
and `mantisv2_results_datasets_7_12.csv`.

## Reproduction issues found (useful for the "lessons learned" section, 30% of the grade)

1. **Dependency clash:** `aeon` (data loader) cannot be imported on current Colab — numba requires NumPy ≤ 2.3, Colab ships NumPy 2.4. Worked around by parsing the `.ts` files directly (no change to data).
2. **figshare blocks scripted downloads:** the TF-C `ndownloader/articles/...` zip links return HTTP 202 with an empty body (bot check) to `wget`. The official figshare REST API (`api.figshare.com/v2/articles/<id>`) gives direct per-file links that work.
3. **Unreported split randomisation for CAP-ID:** paper says "merge all domains, 63.75% / 11.25% / 25% split" but gives no seed or stratification. We used a plain random permutation (seed 0). Sizes match; exact samples may not.
4. **Paper's dataset table vs. TF-C files:** shapes for UCI-HAR and Epilepsy match the paper exactly (3 channels × 206 for UCI-HAR), so the paper's "preprocessed following Zhang et al. (2022)" means the TF-C figshare release.
5. **Compute:** CAP is 19 channels × 3000 steps. Embedding all 40k samples took ~2 min on a T4, but the Random Forest on 25k × 9,728 features is slow on free Colab (2 CPU cores); the Colab UI showed "Connecting" while both cores were busy.
6. **Default classifier variance:** with only 100 test samples, FingerMovements swings ±1 sample per seed (0.54–0.56), so the paper's 0.55 ± 0.01 is basically chance level (binary task) — worth discussing.

# Kaggle runs

Every experiment in this project was a Kaggle notebook. Each folder holds one executed notebook (`.ipynb` with all
outputs, plus `.html` and `.pdf` copies) and the result CSVs it wrote. The folders are numbered in the order the runs
were made. Nothing here is edited by hand.

`python src/tables/build_all_results.py` reads these CSVs and writes the final numbers to
[`results/all_results.csv`](../results/all_results.csv), with a `source_run` column naming the folder each number
came from. If a dataset and method appear in two runs, the later run wins.

| Run | Datasets | What it ran | Code | Hardware | Time | Still used in the final results |
|---|---|---|---|---|---|---|
| [01_reproduction_datasets_7-12](01_reproduction_datasets_7-12/) | 7–12 | Paper protocol: frozen MantisV2 + Random Forest | `src/reproduce_mantisv2.py` | CPU | 2.0 h | RF for 7–12 |
| [02_ideas_1_2_first_try](02_ideas_1_2_first_try/) | 7, 8, 11, 12 | First versions of the mixer and the learned pooling (they overfit) + logistic regression | `src/development/1_ideas_1_2_first_try.py` | T4 GPU | 36 min | LogReg for 7, 8, 11, 12 |
| [03_mixer_v2](03_mixer_v2/) | 7–12 | Plain head and the fixed mixer v2 (idea 1) | `src/development/2_mixer_v2.py` | T4 GPU | 9 min | Head, Mixer for 7–12 |
| [04_idea2_cap_and_lora](04_idea2_cap_and_lora/) | 7–12 | Learned pooling (idea 2) on CAP; LoRA (idea 3) on all six | `src/development/3_idea2_cap_and_lora.py` | T4 GPU | 1.2 h | Pool, Pool+mixer for CAP; LoRA for 7–12 |
| [05_part1_seven_datasets](05_part1_seven_datasets/) | 2–6, 17, 18 | Kit, all 7 methods | `src/run_kaggle_parts.py`, `PART = 1` | T4 GPU | 48 min | everything except the parts re-run in 09 |
| [06_part2_PCL_failed_out_of_RAM](06_part2_PCL_failed_out_of_RAM/) | 13, 14 | Kit on PCL; stopped when the token array ran out of RAM | `PART = 2` | T4 GPU | 25 min | RF, LogReg, Head, Mixer for PCL-ID |
| [07_part3_SEDFx-ID](07_part3_SEDFx-ID/) | 15 | Kit, all 7 methods | `PART = 3` | T4 GPU | 2.0 h | all |
| [08_part4_SEDFx-OOD](08_part4_SEDFx-OOD/) | 16 | Kit, all 7 methods | `PART = 4` | T4 GPU | 1.8 h | all |
| [09_part5_reruns](09_part5_reruns/) | 2, 6, 8, 12, 13, 14 | Re-runs after two memory fixes: all of PCL-OOD, idea 2 + LoRA for PCL-ID, LoRA for EMOPain and MP50, idea 2 for Blink and FingerMovements | `PART = 5` | T4 GPU | 3.3 h | all |
| [10_idea2_UCI-HAR_Epilepsy](10_idea2_UCI-HAR_Epilepsy/) | 7, 11 | Idea 2 with the kit for UCI-HAR and Epilepsy-EEG (replaces the first version from run 02) | `src/mantisv2_kit.py` | T4 GPU | – (notebook not saved, CSVs only) | Pool, Pool+mixer for 7, 11 |

Dataset numbers are the rows of Tables 2 and 3 of the paper (1 Ego4D, 2 EMOPain, 3–4 HHAR, 5–6 MP8/MP50, 7 UCI-HAR,
8 Blink, 9–10 CAP, 11 Epilepsy-EEG, 12 FingerMovements, 13–14 PCL, 15–16 SEDFx, 17–18 SelfRegulationSCP1/2).
Ego4D was not run (licence). Runs 05–08 ran at the same time; the timing of every step is in
[`results/TIMING.md`](../results/TIMING.md).

Runs 01–04 came first and used their own scripts, so their CSVs have different columns; runs 05–10 use the kit
(`src/mantisv2_kit.py`), which writes one `<dataset>_results.csv` per dataset. Both give the same protocol: frozen
MantisV2 layer 2, "combined" token, 3 seeds.

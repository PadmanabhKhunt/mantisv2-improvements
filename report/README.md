# Report (draft)

IEEE double-column conference format, as required by the project specification:
6 pages of content + 1 page work plan + references.

- `main.tex`: the report source. Lines marked `TODO (group)` must be filled in before submission
  (surnames, student numbers, each member's final contribution, and an AI-use statement if the module requires one).
- `main.pdf`: the compiled draft.
- `figures/`: the two figures, built from `results/all_results.csv` by `python src/tables/make_figures.py`.
- `sessions_table.tex`, `timing_table.tex`: the two timing tables (`\input` by `main.tex`), built by
  `python src/tables/make_timing_table.py`.

Every number in the report comes from `results/all_results.csv` (built from the runs in `kaggle_runs/` by
`src/tables/build_all_results.py`), through `results/RESULTS.md` and `results/TIMING.md`.

## Build

On Overleaf: upload `main.tex`, `sessions_table.tex`, `timing_table.tex` and the `figures/` folder, and compile with pdfLaTeX (`IEEEtran` is built in).

Locally (MiKTeX or TeX Live):

```bash
cd report
pdflatex main.tex
pdflatex main.tex
```

# Report (draft)

IEEE double-column conference format, as required by the project specification:
6 pages of content + 1 page work plan + references.

- `main.tex`: the report source. Lines marked `TODO (group)` must be filled in before submission
  (surnames, student numbers, each member's final contribution, and an AI-use statement if the module requires one).
- `main.pdf`: the compiled draft.
- `figures/`: the two figures, built from the result CSVs by `python report/make_figures.py`.

Every number in the report comes from `results/RESULTS.md` (built by `kit/make_results_tables.py`) or from the
`time_s` and `layer_weights` columns of the CSVs in `results/`.

## Build

On Overleaf: upload `main.tex` and the `figures/` folder, and compile with pdfLaTeX (`IEEEtran` is built in).

Locally (MiKTeX or TeX Live):

```bash
cd report
pdflatex main.tex
pdflatex main.tex
```

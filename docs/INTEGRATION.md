# Integration and testing instructions (integration lead)

## What the integration package contains

| Path | Purpose |
|---|---|
| `README.md`, `requirements.txt`, `.gitignore` | project root files |
| `src/__init__.py`, `src/<module>/__init__.py` (all five) | make `src` an importable package; every module folder exists from day one, so feature branches never conflict on `__init__.py` |
| `src/common/config.py`, `io.py`, `geo.py` | shared configuration, raw-file readers, geodesy |
| `scripts/run_first_phase_demo.py` | end-to-end demo; works once all four branches are merged |
| `tests/__init__.py`, `tests/conftest.py`, `tests/test_common.py` | synthetic test trip shared by all test files |
| `data/raw/.gitkeep` | placeholder for the local dataset |
| `docs/INTERFACES.md`, `INTEGRATION.md`, `GIT_WORKFLOW.md`, `FIRST_PHASE_SCOPE.md`, `DEFERRED_COMPONENTS.md`, `AI_ASSISTANCE_AND_WORK_LOG.md` | team documents |

Teammate ZIPs contain **only** their own module files, their own test file and
`docs/team/<Name>_README.md`. No two ZIPs contain the same path, so copying
them into the repository can never overwrite another person's file.

## Merge order

See `docs/GIT_WORKFLOW.md`. Short version: skeleton → preprocessing and data exploration → features and visualisation → evidence.

## Integration test checklist (after each merge, on `main`)

```bash
git checkout main && git pull origin main
```
```bash
pip install -r requirements.txt
```
```bash
python -m pytest -q
```

Expected after all merges: **27 tests pass** (common 7, data exploration 4, preprocessing 5, features 6, visualisation 5).
After intermediate merges, only the test files present on `main` run.

Then run the real-data pipeline (needs the dataset in `data/raw/`):

```bash
python -m src.data_exploration.run_eda
```
```bash
python -m src.preprocessing.run_preprocessing --partition train
```
```bash
python -m src.features.run_features --partition student_test
```
```bash
python -m src.visualization.run_visualizations figures
```
```bash
python scripts/run_first_phase_demo.py --partition train --trip 00000010
```

Sanity values to compare against (from our own run on 18 Nov 2026, see the work log):

| Check | Expected |
|---|---|
| `outputs/eda/inventory_summary.csv` | train 80 trips / 208,417 GPS rows; student_test 10 trips / 19,160 rows |
| `validation_issues.csv` | 181 rows; no `error`-severity issues |
| preprocessing totals (train) | 0 rows dropped; 133 gaps > 30 s; 68 speed-spike flags |
| `feature_summary.csv` (student_test) | observation-weighted candidate miss rate 0.0026 at 50 m |

If your numbers differ, the code or the data differs. Find out why before the evaluation. Don't edit the documents to match.

## Static checks done before packaging

* `python -m py_compile` on every `.py` file: passes
* Every import uses the `src.` prefix and respects the dependency direction in `docs/INTERFACES.md`
* No file path appears in more than one ZIP
* Each ZIP was re-extracted and the full test suite passed on the re-assembled repository

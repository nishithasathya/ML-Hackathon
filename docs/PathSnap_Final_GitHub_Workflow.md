# PathSnap final evaluation: GitHub workflow

This is the same process as Evaluation 1. There is one repository (`https://github.com/nishithasathya/ML-Hackathon`),
one `main` branch, one feature branch per work package, and pull requests that Nishitha merges.
**Each person copies only their own ZIP and commits under their own GitHub account.**
The phase-1 skeleton is already merged, so **no new skeleton branch** is created.

## Packages

| ZIP | Owner | Branch(es) |
|---|---|---|
| `PathSnap_Final_Integration_Package.zip` | Nishitha | `integration/final-shared` |
| `Neha_Final.zip` | Neha | `feature/model-training` |
| `Nishanth_Final.zip` | Nishanth | `feature/features-hmm`, then `feature/bayesian-optimisation` |
| `Niranjan_Final.zip` | Niranjan | `feature/api-ui-deployment` |
| `Nishitha_Final.zip` | Nishitha | `feature/inference-pipeline`, then `results/final-evidence` |

`PathSnap_Final_Integrated_Project.zip` is the complete merged result, for reference and for checking. **Do not push it.**

## Merge order (each round needs the previous ones merged)

| Round | Branch | Who | Imports from |
|---|---|---|---|
| 1 | `integration/final-shared` | Nishitha | phase-1 code only |
| 2 | `feature/features-hmm` | Nishanth | round 1 (`src/common/graph.py`) |
| 3 | `feature/model-training` | Neha | rounds 1-2 (features, evaluation) |
| 4 | `feature/bayesian-optimisation` | Nishanth | round 3 (`src/models`) |
| 5 | `feature/inference-pipeline` | Nishitha | rounds 2-3 (features, hmm, preprocessing) |
| 6 | `feature/api-ui-deployment` | Niranjan | round 5 (`src/inference`) |
| 7 | `results/final-evidence` | Nishitha | everything (full run, artefacts, report) |

Everyone can copy files and commit **immediately**. Before opening the PR, run
`git fetch origin && git merge origin/main` once the previous round is merged, then run the tests.

## One-time setup (everyone, on their own laptop)

```bash
git config --global user.name "Your Full Name"
```
```bash
git config --global user.email "the-email-of-your-github-account"
```

If you don't have the repo yet:

```bash
cd ~/Documents && git clone https://github.com/nishithasathya/ML-Hackathon.git
```

Then:

```bash
cd ~/Documents/ML-Hackathon
```
```bash
python3 -m venv .venv && source .venv/bin/activate
```

Unzip your package in Downloads (double-click it, or run `unzip -o ~/Downloads/<ZIP>.zip -d ~/Downloads/`).
Every ZIP unpacks to a folder with the same name. Each ZIP has a `START_HERE.md`: read it, but do not copy it into the repo.

---

## Round 1: Nishitha, `integration/final-shared`

```bash
cd ~/Documents/ML-Hackathon && git checkout main && git pull origin main
```
```bash
git checkout -b integration/final-shared
```
```bash
cd ~/Downloads/PathSnap_Final_Integration_Package && cp -R src tests docs requirements.txt .gitignore .env.example README.md ~/Documents/ML-Hackathon/ && cd ~/Documents/ML-Hackathon
```
```bash
pip install -r requirements.txt
```
```bash
python -m pytest -q tests/test_common.py tests/test_evaluation.py
```
```bash
git add src/common src/evaluation tests/conftest.py tests/test_evaluation.py requirements.txt .gitignore .env.example README.md docs/FINAL_AUDIT.md docs/INTERFACES_FINAL.md docs/DEFERRED_COMPONENTS.md docs/PathSnap_Final_GitHub_Workflow.md docs/team/Nishitha_FINAL_README.md
```
```bash
git status
```
```bash
git commit -m "Final shared layer: eval-data config, label-free loader, road graph, evaluation metrics"
```
```bash
git push -u origin integration/final-shared
```

Open a PR to `main`. Ask a teammate to review it, then merge with **Create a merge commit**.

## Round 2: Nishanth, `feature/features-hmm`

```bash
cd ~/Documents/ML-Hackathon && git checkout main && git pull origin main
```
```bash
git checkout -b feature/features-hmm
```
```bash
cp -R ~/Downloads/Nishanth_Final/round2_features_hmm/src ~/Downloads/Nishanth_Final/round2_features_hmm/tests ~/Downloads/Nishanth_Final/round2_features_hmm/docs .
```
```bash
pip install -r requirements.txt
```
```bash
python -m pytest -q tests/test_features.py tests/test_hmm.py
```
```bash
git add src/features tests/test_features.py src/hmm tests/test_hmm.py docs/team/Nishanth_FINAL_README.md
```
```bash
git commit -m "Context features, duplicate-arc labels, radius study; HMM transitions/emissions and Viterbi decoder"
```
```bash
git push -u origin feature/features-hmm
```

## Round 3: Neha, `feature/model-training`

```bash
cd ~/Documents/ML-Hackathon && git checkout main && git pull origin main
```
```bash
git checkout -b feature/model-training
```
```bash
cp -R ~/Downloads/Neha_Final/src ~/Downloads/Neha_Final/tests ~/Downloads/Neha_Final/docs .
```
```bash
pip install -r requirements.txt
```
```bash
python -m pytest -q tests/test_preprocessing.py tests/test_models.py
```
```bash
python -m src.models.train_compare
```

The `train_compare` run needs the dataset in `data/raw/` and takes about 25 minutes. Running it on your own laptop is how you get your own logs.

```bash
git add src/models tests/test_models.py docs/team/Neha_FINAL_README.md
```
```bash
git commit -m "Five candidate-scoring models with trip-grouped CV, comparison and selection rule"
```
```bash
git push -u origin feature/model-training
```

## Round 4: Nishanth, `feature/bayesian-optimisation`

```bash
git checkout main && git pull origin main && git checkout -b feature/bayesian-optimisation
```
```bash
cp -R ~/Downloads/Nishanth_Final/round4_optimisation/src ~/Downloads/Nishanth_Final/round4_optimisation/tests .
```
```bash
python -m pytest -q tests/test_optimization.py
```
```bash
git add src/optimization tests/test_optimization.py
```
```bash
git commit -m "Optuna TPE Bayesian optimisation with group CV objective and tuned-vs-untuned comparison"
```
```bash
git push -u origin feature/bayesian-optimisation
```

## Round 5: Nishitha, `feature/inference-pipeline`

```bash
git checkout main && git pull origin main && git checkout -b feature/inference-pipeline
```
```bash
cp -R ~/Downloads/Nishitha_Final/round5_inference/src ~/Downloads/Nishitha_Final/round5_inference/scripts ~/Downloads/Nishitha_Final/round5_inference/tests .
```
```bash
python -m pytest -q tests/test_inference.py
```
```bash
git add src/inference scripts tests/test_inference.py
```
```bash
git commit -m "Artefact bundle export/validation, label-free trip predictor, pipeline and rehearsal scripts"
```
```bash
git push -u origin feature/inference-pipeline
```

## Round 6: Niranjan, `feature/api-ui-deployment`

```bash
cd ~/Documents/ML-Hackathon && git checkout main && git pull origin main
```
```bash
git checkout -b feature/api-ui-deployment
```
```bash
cp -R ~/Downloads/Niranjan_Final/src ~/Downloads/Niranjan_Final/frontend ~/Downloads/Niranjan_Final/deployment ~/Downloads/Niranjan_Final/tests ~/Downloads/Niranjan_Final/docs .
```
```bash
pip install -r requirements.txt
```
```bash
python -m pytest -q tests/test_api.py tests/test_ui.py tests/test_visualization.py
```
```bash
git add src/api frontend src/visualization deployment tests/test_api.py tests/test_ui.py tests/test_visualization.py docs/team/Niranjan_FINAL_README.md
```
```bash
git commit -m "FastAPI backend with server-side evaluator and request logs, Streamlit evaluator UI, maps, deployment kit"
```
```bash
git push -u origin feature/api-ui-deployment
```

## Round 7: Nishitha, `results/final-evidence`

Everything is merged. Run the whole system on your own laptop and commit the evidence.

```bash
git checkout main && git pull origin main && git checkout -b results/final-evidence
```
```bash
python -m pytest -q
```

Either run `./scripts/run_full_pipeline.sh` (about 2 hours), or copy the evidence produced on 18 Nov 2026:

```bash
cp -R ~/Downloads/Nishitha_Final/round7_evidence/results ~/Downloads/Nishitha_Final/round7_evidence/artifacts ~/Downloads/Nishitha_Final/round7_evidence/docs .
```
```bash
python scripts/validate_artifacts.py --trip-dir data/raw/student_test --trip 00000052
```
```bash
git add results artifacts docs/technical_report docs/AI_ASSISTANCE_AND_WORK_LOG.md
```
```bash
git commit -m "Final experiment results, exported model bundle and technical report"
```
```bash
git push -u origin results/final-evidence
```

`artifacts/model.joblib` is about 50–90 MB. GitHub warns above 50 MB and rejects files above 100 MB.
If the push is rejected, publish the file as a GitHub Release asset instead: on the repo page go to
Releases → Draft new release → attach `model.joblib`. Then write the download URL in `artifacts/README.md`
and remove the file from git with `git rm --cached artifacts/model.joblib`.

## After every merge (Nishitha)

```bash
git checkout main && git pull origin main && python -m pytest -q
```

Then tell the next person to run:

```bash
git fetch origin && git merge origin/main
```

## Never commit

`.venv/`, `.env`, `data/raw/*` (the dataset), `data/processed/`, `outputs/`, `logs/`, `__pycache__/`.
These are already in `.gitignore`. Run `git status` before every commit.

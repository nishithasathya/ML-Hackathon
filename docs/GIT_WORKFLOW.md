# Git workflow: exact commands for each teammate

Repository: `https://github.com/nishithasathya/ML-Hackathon.git`

Rules:
* Nobody commits directly to `main`. Every change arrives through a pull request (PR).
* Each person only copies **their own** ZIP's files. Never copy another teammate's files.
* The integration lead (Nishitha) reviews and merges PRs in the order below.
* Commit under your own name. This matters because the repository history is part of the evaluation.

The commands assume macOS/Linux and that the ZIP was downloaded to `~/Downloads`.
Adjust paths if yours differ. Run them one at a time.

## One-time setup (everyone)

```bash
git config --global user.name "Your Full Name"
```
```bash
git config --global user.email "the-email-on-your-github-account"
```
```bash
cd ~/Documents
```
```bash
git clone https://github.com/nishithasathya/ML-Hackathon.git
```
```bash
cd ML-Hackathon
```

Nishitha must add the other three as collaborators first: GitHub, then Settings → Collaborators → Add people.

## Merge order

| Round | Branch | Who | Depends on |
|---|---|---|---|
| 0 | `setup/project-skeleton` | Nishitha | nothing |
| 1 | `feature/preprocessing` | Neha | round 0 |
| 1 | `feature/data-exploration` | Nishitha | round 0 |
| 2 | `feature/feature-engineering` | Nishanth | rounds 0-1 (imports preprocessing) |
| 2 | `feature/visualization-documentation` | Niranjan | rounds 0-1 (imports preprocessing) |
| 3 | `docs/phase1-evidence` | Niranjan | everything merged |

People in round 2 can start writing and reading code immediately. They only need
`git merge origin/main` (shown below) before opening their PR.

---

## Round 0: Nishitha: shared skeleton (Integration package)

```bash
cd ~/Documents/ML-Hackathon
```
```bash
git checkout main && git pull origin main
```
```bash
git checkout -b setup/project-skeleton
```
```bash
unzip -o ~/Downloads/Integration_Package_First_Evaluation.zip -d ~/Downloads/
```
```bash
cp -R ~/Downloads/Integration_Package_First_Evaluation/. .
```
```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```
```bash
python -m pytest -q tests/test_common.py
```
```bash
git add -A && git status
```
```bash
git commit -m "Add shared project skeleton: config, raw-data I/O, geodesy, docs, test fixtures"
```
```bash
git push -u origin setup/project-skeleton
```

Open a PR on GitHub (`setup/project-skeleton` → `main`) and merge it.
`git status` must **not** list `data/raw/train` or `outputs/`; `.gitignore` excludes them.

## Round 1a: Neha: preprocessing

```bash
cd ~/Documents/ML-Hackathon && git checkout main && git pull origin main
```
```bash
git checkout -b feature/preprocessing
```
```bash
unzip -o ~/Downloads/Neha_First_Evaluation.zip -d ~/Downloads/
```
```bash
cp -R ~/Downloads/Neha_First_Evaluation/src ~/Downloads/Neha_First_Evaluation/tests ~/Downloads/Neha_First_Evaluation/docs .
```
```bash
python -m pytest -q tests/test_common.py tests/test_preprocessing.py
```
```bash
python -m src.preprocessing.run_preprocessing --partition train
```
```bash
git add src/preprocessing tests/test_preprocessing.py docs/team/Neha_README.md
```
```bash
git commit -m "Add GPS track preprocessing: cleaning, time ordering, gap segments, spike flags"
```
```bash
git push -u origin feature/preprocessing
```

Open a PR `feature/preprocessing` → `main`.

## Round 1b: Nishitha: data exploration

```bash
cd ~/Documents/ML-Hackathon && git checkout main && git pull origin main
```
```bash
git checkout -b feature/data-exploration
```
```bash
unzip -o ~/Downloads/Nishitha_First_Evaluation.zip -d ~/Downloads/
```
```bash
cp -R ~/Downloads/Nishitha_First_Evaluation/src ~/Downloads/Nishitha_First_Evaluation/tests ~/Downloads/Nishitha_First_Evaluation/docs .
```
```bash
python -m pytest -q tests/test_common.py tests/test_data_exploration.py
```
```bash
python -m src.data_exploration.run_eda
```
```bash
git add src/data_exploration tests/test_data_exploration.py docs/team/Nishitha_README.md
```
```bash
git commit -m "Add dataset inventory, validation and per-trip statistics (EDA)"
```
```bash
git push -u origin feature/data-exploration
```

Open a PR. Ask a teammate to review it, since you are also the one who merges.

## Round 2a: Nishanth: feature engineering

```bash
cd ~/Documents/ML-Hackathon && git checkout main && git pull origin main
```
```bash
git checkout -b feature/feature-engineering
```
```bash
unzip -o ~/Downloads/Nishanth_First_Evaluation.zip -d ~/Downloads/
```
```bash
cp -R ~/Downloads/Nishanth_First_Evaluation/src ~/Downloads/Nishanth_First_Evaluation/tests ~/Downloads/Nishanth_First_Evaluation/docs .
```
If preprocessing was not merged yet when you branched, bring it in once it is:
```bash
git fetch origin && git merge origin/main
```
```bash
python -m pytest -q tests/test_features.py
```
```bash
python -m src.features.run_features --partition train --sweep 25 50 75 100
```
```bash
git add src/features tests/test_features.py docs/team/Nishanth_README.md
```
```bash
git commit -m "Add candidate generation, route-to-observation labels and initial features"
```
```bash
git push -u origin feature/feature-engineering
```

## Round 2b: Niranjan: visualisation and documentation

```bash
cd ~/Documents/ML-Hackathon && git checkout main && git pull origin main
```
```bash
git checkout -b feature/visualization-documentation
```
```bash
unzip -o ~/Downloads/Niranjan_First_Evaluation.zip -d ~/Downloads/
```
```bash
cp -R ~/Downloads/Niranjan_First_Evaluation/src ~/Downloads/Niranjan_First_Evaluation/tests ~/Downloads/Niranjan_First_Evaluation/docs .
```
```bash
git fetch origin && git merge origin/main
```
```bash
python -m pytest -q tests/test_visualization.py
```
```bash
git add src/visualization tests/test_visualization.py docs/FIRST_PHASE_TECHNICAL_DOC.md docs/SETUP.md docs/DEMO_CHECKLIST.md docs/team/Niranjan_README.md
```
```bash
git commit -m "Add Folium maps, EDA figures and first-phase documentation"
```
```bash
git push -u origin feature/visualization-documentation
```

## Round 3: Niranjan: generate and commit evidence (after everything is merged)

```bash
git checkout main && git pull origin main && git checkout -b docs/phase1-evidence
```
```bash
python -m pytest -q
```
```bash
python -m src.data_exploration.run_eda && python -m src.features.run_features --partition train --sweep 25 50 75 100
```
```bash
python -m src.visualization.run_visualizations figures && python -m src.visualization.run_visualizations maps --partition train --trips 00000010 00000033
```
```bash
python -m src.visualization.run_visualizations maps --partition student_test --trips 00000052
```
```bash
mkdir -p docs/evidence && cp outputs/eda/data_quality_notes.md outputs/eda/trip_statistics.csv outputs/eda/inventory_summary.csv outputs/features/train/radius_sweep.csv docs/evidence/
```
```bash
cp outputs/figures/*.png outputs/maps/student_test_00000052.html outputs/maps/train_00000010.html docs/evidence/
```
```bash
git add docs/evidence && git commit -m "Add first-phase evidence generated from the current code" && git push -u origin docs/phase1-evidence
```

## Reviewing and merging (Nishitha)

1. On the PR page, check **Files changed**. The PR must only touch the owner's folders (see `docs/INTERFACES.md`).
2. Check out the branch locally and run the tests:
   ```bash
   git fetch origin && git checkout feature/preprocessing && python -m pytest -q
   ```
3. Merge with **"Create a merge commit"** so that each author's commits stay visible in the history.
4. Tell everyone to update their branches:
   ```bash
   git fetch origin && git merge origin/main
   ```

## If something goes wrong

* **Merge conflict:** you edited a file you don't own. Keep the owner's version (`git checkout --theirs <file>` while merging origin/main), then talk to the owner.
* **Accidentally committed data/outputs:**
  ```bash
  git rm -r --cached data/raw outputs && git commit -m "Untrack data and outputs"
  ```
* **Pushed to main by mistake:** tell Nishitha. Don't force-push.

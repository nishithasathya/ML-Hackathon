# First-evaluation demonstration checklist

## The day before

- [ ] All PRs merged; `main` passes `python -m pytest -q` (27 passed) on **at least two** team laptops
- [ ] Dataset copied to `data/raw/` on the demo laptop (`ls data/raw/train | wc -l` → 80)
- [ ] Full pipeline run once on the demo laptop (commands in README), outputs present in `outputs/`
- [ ] `docs/evidence/` committed (Round 3 in `GIT_WORKFLOW.md`)
- [ ] Open the Folium maps once with internet on, so the map tiles and Leaflet load
- [ ] Repository is public, or the evaluators have access

## Live demo script (≈ 8 minutes)

| # | Who | Show | Command / file |
|---|---|---|---|
| 1 | Nishitha | Repo structure, branches and PR history on GitHub | GitHub → Insights → Network |
| 2 | Nishitha | Dataset inventory + data-quality notes | `python -m src.data_exploration.run_eda` → `outputs/eda/data_quality_notes.md` |
| 3 | Neha | Cleaning report: gaps, segments, spike flags; explain `obs_idx` and why spikes are flagged, not deleted | `python -m src.preprocessing.run_preprocessing --partition student_test` |
| 4 | Nishanth | Candidate generation + radius sweep; nearest-arc vs twin-arc finding; heading feature | `outputs/figures/radius_sweep.png`, `feature_summary.csv` |
| 5 | Niranjan | Folium map: toggle layers, show the offset trip 00000033 and a clean trip | `outputs/maps/train_00000033.html`, `student_test_00000052.html` |
| 6 | Anyone | End-to-end demo on a trip chosen by the evaluator (train or student_test) | `python scripts/run_first_phase_demo.py --partition train --trip <id>` |
| 7 | Anyone | Tests | `python -m pytest -q` |

## Questions each member should be able to answer

**Everyone**
- What do `.track`, `.nodes`, `.arcs`, `.route` contain, and how do they relate? (row index = node id / arc id; per-trip network)
- What is implemented, and what is deliberately not? (`DEFERRED_COMPONENTS.md`)
- How do we avoid leakage? (partition guard; route only used for labels; radius chosen on train)

**Nishitha (exploration/integration)**
- Why is the equirectangular projection acceptable here, and what was its measured error? (≤ 0.78 %)
- What are weak vs strong components, and why do they matter for an HMM?
- Which data-quality problems did you find? (§4 of the technical doc)

**Neha (preprocessing)**
- Why 30 s for the gap threshold and 70 m/s for spikes? Why must both "speed in" and "speed out" exceed it?
- Why are arcs not de-duplicated? (ids must stay aligned with `.route`)
- What happens to the `obs_idx` of a dropped row?

**Nishanth (features)**
- Prove that the KD-tree search with radius r + spacing/2 never misses an arc.
- Explain the dynamic-programming alignment and why nearest-route-arc is not enough.
- Why is `is_reference` never a feature? Why does `heading_diff_deg` matter?

**Niranjan (visualisation/docs)**
- What layers does the map have, and why is only the nearby network drawn?
- Which partition does each figure use?
- Where does each number in the technical document come from?

## Do NOT

- Claim any model, accuracy, F1, HMM or API results. None exist yet.
- Show or mention evaluator-testing data.
- Edit numbers in the documentation by hand.

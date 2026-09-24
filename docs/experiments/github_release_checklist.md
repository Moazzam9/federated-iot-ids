# GitHub Release Pre-Flight Checklist — v1.0.0

**Repository:** `federated-iot-ids`  
**Target Tag:** `v1.0.0`  
**Prepared by:** Stage 13 — Final GitHub Release Preparation

Complete all items before creating the GitHub Release.

---

## 1. Repository Integrity

- [x] Working tree is clean (`git status` shows no uncommitted changes)
- [x] All changes pushed to `origin/main` (`git push origin main` confirmed)
- [x] Latest commit hash recorded: see `git log -1 --oneline`
- [x] No accidentally tracked large files (dataset CSVs, model checkpoints, `.venv/`)
- [x] No secrets or credentials found in tracked files (`git grep` scan passed)
- [x] `.gitignore` correctly excludes `data/raw/`, `results/raw/`, `.venv/`, `__pycache__/`, `build/`
- [x] `.gitignore` correctly includes publication figures, tables, and `figures/`

---

## 2. Manuscript

- [x] `manuscript/manuscript.md` is complete (all 8 sections present)
- [x] Abstract accurately reflects empirical results (F1 scores, scope)
- [x] No unfinished placeholders (`TODO`, `FIXME`, `TBD`, `PLACEHOLDER`, `XXX`)
- [x] No internal reference phrases (e.g., "Publication 02") remaining in any file
- [x] All 15 references (`[1]`–`[15]`) are verified and cross-cited
- [x] Publication status language does not claim peer review, acceptance, or DOI
- [x] Manuscript does not claim unmeasured properties (energy, physical deployment, formal DP)

---

## 3. Scientific Results

- [x] Centralized F1 = 0.997097 verified against `results/raw/final_test_evaluation/final_test_evaluation.json`
- [x] IID FedAvg F1 = 0.999430 verified against raw JSON
- [x] Device Non-IID FedAvg F1 = 0.975696 verified against raw JSON (typo 0.975796 not present)
- [x] Local-Only Macro F1 = 0.971422 verified against raw JSON
- [x] Communication payload = 38,148 bytes/transfer; 2,059,992 bytes cumulative — verified
- [x] All manuscript values consistent with `results/processed/publication/tables/` CSVs
- [x] `docs/experiments/final_publication_package_manifest.md` headline metrics table is correct

---

## 4. Reproducibility Package

- [x] `docs/experiments/reproducibility.md` present and complete (19 sections)
- [x] All experiment entry-point scripts exist under `experiments/scripts/`
- [x] `configs/experiment.yaml` present with `random_seed: 42`
- [x] `data/processed/splits/split_specification.txt` present (frozen split spec)
- [x] `data/processed/preprocessing/standard_scaler.pkl` present (persisted scaler)
- [x] `requirements.txt` present with verified dependency list
- [x] Virtual environment instructions included in reproducibility guide

---

## 5. Publication Figures and Tables

- [x] 6 PNG figures present in `figures/` (top-level)
- [x] 6 PNG figures present in `results/processed/publication/figures/`
- [x] 6 CSV tables present in `results/processed/publication/tables/`
- [x] All figures referenced correctly in `manuscript/manuscript.md`
- [x] All tables embedded or referenced correctly in manuscript

---

## 6. Tests

- [x] `pytest --basetemp=build/pytest_tmp -q` passes with **105 passed, 0 failed**
- [x] No test imports or fixtures depend on the N-BaIoT dataset being present
- [x] `tests/test_reproducibility_artifacts.py` verifies config YAML, split spec, scaler, and SmallMLP parameter count

---

## 7. Documentation and Metadata

- [x] `README.md` present with project identity, study summary, structure, and reproducibility pointer
- [x] `CITATION.cff` present with software citation metadata (no DOI — not yet published)
- [x] `docs/experiments/github_release_notes.md` complete
- [x] `docs/experiments/github_release_checklist.md` complete (this file)
- [x] `docs/experiments/scientific_consistency_audit.md` complete (29-area audit)
- [x] `docs/experiments/literature_audit.md` complete (15 references verified)

---

## 8. License

- [x] **LICENSE file present** — MIT License added to repository root.

---

## 9. Final Pre-Release Actions (to be done at release time)

- [ ] Add `LICENSE` file (decision required — see Section 8)
- [ ] Create and push Git tag: `git tag -a v1.0.0 -m "Release v1.0.0"` then `git push origin v1.0.0`
- [ ] Create GitHub Release via GitHub web interface using tag `v1.0.0` and content from `docs/experiments/github_release_notes.md`
- [ ] (Optional) Upload release assets if required (e.g., manuscript PDF, zipped reproducibility package)
- [ ] (Optional) Register with Zenodo for DOI assignment after publication acceptance

---

## Status

**Stage 13 preparation is complete. All pre-release checks are satisfied.**

The repository is ready for the GitHub Release to be created using tag `v1.0.0`.

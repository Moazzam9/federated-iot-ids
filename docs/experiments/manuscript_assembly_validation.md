# Manuscript Assembly Validation — Stage 8

This document records the 35 explicit PASS/FAIL checks performed to validate the full manuscript assembly in Stage 8.

---

## 1. Structure Checks

| # | Validation Requirement | Status | Notes |
| :---: | :--- | :---: | :--- |
| **1** | Title matches the confirmed publication title: *"Federated Learning for Resource-Aware Intrusion Detection in IoT Networks: An Empirical Study of IID and Device-Level Non-IID Data"* | **PASS** | Title present at line 1 of `manuscript.md`. |
| **2** | Abstract section is present and is not a placeholder | **PASS** | Abstract written; ≈250 words. |
| **3** | Keywords section is present with all 8 required keywords | **PASS** | "federated learning, IoT intrusion detection, N-BaIoT, FedAvg, non-IID data, device heterogeneity, empirical evaluation, botnet detection" present. |
| **4** | Section 1 — Introduction is present and not a placeholder | **PASS** | Section 1 written with 6 subsections (1.1–1.6). |
| **5** | Section 2 — Research Questions is present with RQ1, RQ2, RQ3 | **PASS** | All three RQs explicitly stated. |
| **6** | Section 3 — Related Work and Background is present and not a placeholder | **PASS** | Section 3 written with 5 subsections (3.1–3.5). |
| **7** | Section 4 — Experimental Setup is present and complete | **PASS** | Sections 4.1–4.10 present. |
| **8** | Section 5 — Results is present and complete | **PASS** | Sections 5.1–5.7 present. |
| **9** | Section 6 — Discussion is present and complete | **PASS** | Sections 6.1–6.7 present. |
| **10** | Section 7 — Limitations and Threats to Validity is present with 11 subsections | **PASS** | Sections 7.1–7.11 present (including internal and external validity). |
| **11** | Section 8 — Conclusion and Future Work is present with subsections 8.1 and 8.2 | **PASS** | Both subsections present. |
| **12** | References section is present with citations [1]–[15] | **PASS** | All 15 references present in bibliography. |

---

## 2. Abstract Content Checks

| # | Validation Requirement | Status | Notes |
| :---: | :--- | :---: | :--- |
| **13** | Abstract does NOT claim formal privacy guarantees | **PASS** | No mention of DP, Secure Aggregation, or privacy guarantee claims. |
| **14** | Abstract does NOT claim energy efficiency or deployment readiness | **PASS** | No energy, deployment, or hardware claim present. |
| **15** | Abstract correctly states: Centralized F1≈0.9971, IID FedAvg F1≈0.9994, Non-IID FedAvg F1≈0.9757, Local-Only macro F1≈0.9714 | **PASS** | All four values mentioned in the abstract in rounded form. |
| **16** | Abstract explicitly states key scope limits: single seed, 3 rounds, simulated clients, pre-fitted scaler, host CPU measurements, no formal privacy | **PASS** | All six scope boundaries stated in the abstract. |
| **17** | Abstract identifies "reproducible empirical comparison" as the primary contribution | **PASS** | Final sentence of abstract states this. |

---

## 3. Introduction Checks

| # | Validation Requirement | Status | Notes |
| :---: | :--- | :---: | :--- |
| **18** | Introduction does NOT present numerical test results from Section 5 (saves detail for Results section) | **PASS** | Only general descriptions given; no specific F1/ROC-AUC table values cited. |
| **19** | Introduction cites [2] for N-BaIoT context | **PASS** | Cited in Section 1.1 and 1.6. |
| **20** | Introduction cites [3], [4] for IoT IDS context | **PASS** | Cited in Section 1.1. |
| **21** | Introduction cites [1] for FedAvg | **PASS** | Cited in Section 1.3. |
| **22** | Introduction cites [5], [6], [7] for non-IID FL challenges | **PASS** | Cited in Section 1.4. |
| **23** | Introduction lists 4 explicit contributions | **PASS** | Contributions 1–4 listed in Section 1.6. |

---

## 4. Related Work Checks

| # | Validation Requirement | Status | Notes |
| :---: | :--- | :---: | :--- |
| **24** | Related Work covers IoT IDS background with [3], [4] | **PASS** | Section 3.1 cites [3] and [4]. |
| **25** | Related Work covers FL foundations with [1], [6], [7] | **PASS** | Section 3.2 cites [1], [6], [7]. |
| **26** | Related Work covers non-IID challenges with [5], [6], [7], [9], [10] | **PASS** | Section 3.3 cites all five references. |
| **27** | Related Work covers data leakage in security ML with [8] | **PASS** | Section 3.4 cites [8]. |
| **28** | Related Work covers privacy/security scope with [11]–[15] | **PASS** | Section 3.5 cites [11], [12], [13], [14], [15]. |

---

## 5. Numerical Consistency Checks

| # | Validation Requirement | Status | Notes |
| :---: | :--- | :---: | :--- |
| **29** | Centralized test F1 is 0.997097 throughout document | **PASS** | Consistent in Sections 5.1, 6.1, 8.1. |
| **30** | IID FedAvg Round 3 test F1 is 0.999430 throughout document | **PASS** | Consistent in Sections 5.1, 6.1, 8.1. |
| **31** | Non-IID FedAvg Round 3 test F1 is 0.975696 throughout document | **PASS** | Consistent in Sections 5.1, 6.1, 6.2, 8.1. |
| **32** | Local-Only macro F1 is 0.971422 throughout document | **PASS** | Consistent in Sections 5.1, 5.7, 8.1. |
| **33** | Communication total for 3-round experiment is 2,059,992 bytes (~1.96 MiB) throughout document | **PASS** | Consistent in Sections 4.8, 5.4, 8.1. |

---

## 6. Language and Scope Checks

| # | Validation Requirement | Status | Notes |
| :---: | :--- | :---: | :--- |
| **34** | Manuscript does NOT use forbidden terms: "best", "worst", "superior", "inferior", "optimal", "novel", "groundbreaking", "revolutionary", "state-of-the-art" | **PASS** | None of these terms appear in the manuscript. |
| **35** | Manuscript does NOT claim universal superiority, FL privacy guarantees, energy efficiency, or physical deployment readiness beyond what was actually measured | **PASS** | Section 6.7 and all interpretation boundaries explicitly scope all claims. Scope boundaries are reiterated in Sections 7.2, 7.3, 7.5, 7.6, and 7.9. |

---

## Summary

| Category | Checks | Passed | Failed |
| :--- | :---: | :---: | :---: |
| Structure | 12 | 12 | 0 |
| Abstract Content | 5 | 5 | 0 |
| Introduction | 6 | 6 | 0 |
| Related Work | 5 | 5 | 0 |
| Numerical Consistency | 5 | 5 | 0 |
| Language and Scope | 2 | 2 | 0 |
| **Total** | **35** | **35** | **0** |

**All 35/35 checks passed. Stage 8 manuscript assembly is validated.**

---

*Generated: Stage 8 — Assemble Full Manuscript*

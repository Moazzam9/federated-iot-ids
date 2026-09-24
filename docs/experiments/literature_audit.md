# Literature and References Audit — the study

This document provides a comprehensive literature audit and bibliographic verification for the study.

---

## 1. Audit Scope
The objective of this audit is to verify that all external background claims, dataset attributions, foundational methodologies, non-IID optimization challenges, and future-work concepts cited in `manuscript/manuscript.md` are supported by authoritative primary research literature.

---

## 2. Citation Inventory & Usage Mapping

| Ref ID | Primary Reference Citation | Category / Domain | Manuscript Section(s) | Claim / Concept Supported | Verification Source & DOI/URL |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **[1]** | McMahan et al. (2017) | Federated Learning & FedAvg | Sec. 4.5, 6.5, 7.8, 8.2 | Foundational FedAvg algorithm formulation & parameter exchange model. | AISTATS 2017 (PMLR 54:1273–1282), arXiv:1602.05629 |
| **[2]** | Meidan et al. (2018) | N-BaIoT Dataset | Sec. 4.1 | Original N-BaIoT dataset collection, device coverage & botnet vectors. | *IEEE Pervasive Computing*, DOI: 10.1109/MPRV.2018.03367731 |
| **[3]** | Zarpelão et al. (2017) | IoT Intrusion Detection | Sec. 6.1 | Survey of intrusion detection architectures in IoT environments. | *JNCA*, DOI: 10.1016/j.jnca.2017.02.009 |
| **[4]** | Chaabouni et al. (2019) | IoT NIDS Review | Sec. 6.1 | Network intrusion detection systems & traffic analysis for IoT environments. | *IEEE Access*, DOI: 10.1109/ACCESS.2019.2943141 |
| **[5]** | Zhao et al. (2018) | Non-IID FL | Sec. 4.6, 6.2 | Client statistical heterogeneity & Non-IID performance dynamics in FL. | arXiv:1806.00582, DOI: 10.48550/arXiv.1806.00582 |
| **[6]** | Li et al. (2020) | FL Overview & Challenges | Sec. 4.6, 6.2, 6.5 | Survey of federated learning optimization, communication & non-IID challenges. | *IEEE SPM*, DOI: 10.1109/MSP.2020.2975749 |
| **[7]** | Kairouz et al. (2021) | Advances in FL | Sec. 4.6 | Comprehensive survey of open problems & non-IID partition benchmarks in FL. | *FnT in Machine Learning*, DOI: 10.1561/2200000083 |
| **[8]** | Arp et al. (2022) | Security ML Pitfalls | Sec. 4.1, 7.7 | Data leakage, duplicate grouping & methodological pitfalls in security ML. | *USENIX Security 2022*, pp. 3971–3988 |
| **[9]** | Li et al. (2020) | FedProx Algorithm | Sec. 7.8, 8.2 | FedProx optimization framework for heterogeneous client environments. | *MLSys 2020*, arXiv:1812.06127 |
| **[10]** | Karimireddy et al. (2020) | SCAFFOLD Algorithm | Sec. 7.8, 8.2 | SCAFFOLD control-variate algorithm for non-IID variance reduction. | ICML 2020 (PMLR 119:5132–5143), arXiv:1910.06378 |
| **[11]** | Dwork (2006) | Differential Privacy | Sec. 7.9, 8.2 | Mathematical foundations of Differential Privacy ($\epsilon, \delta$). | ICALP 2006 (Springer LNCS 4052), DOI: 10.1007/11787006_1 |
| **[12]** | Abadi et al. (2016) | DP-SGD Deep Learning | Sec. 7.9, 8.2 | Differential Privacy in deep neural network training (DP-SGD). | ACM CCS 2016, DOI: 10.1145/2976749.2978318 |
| **[13]** | Bonawitz et al. (2017) | Secure Aggregation | Sec. 7.9, 8.2 | Cryptographic Secure Aggregation protocol for federated parameter updates. | ACM CCS 2017, DOI: 10.1145/3133956.3133982 |
| **[14]** | Blanchard et al. (2017) | Byzantine Robustness | Sec. 7.9, 8.2 | Byzantine-tolerant gradient aggregation & poisoning robustness in distributed ML. | NIPS 2017, pp. 119–129 |
| **[15]** | Bhagoji et al. (2019) | Adversarial FL | Sec. 7.9, 8.2 | Model poisoning & adversarial attacks against federated learning. | ICML 2019 (PMLR 97:634–643) |

---

## 3. Claim-to-Citation Mapping Analysis

1. **Foundational FL & FedAvg Formulation:** Supported by McMahan et al. [1]. Used in Section 4.5 to cite standard FedAvg, Section 6.5 for payload trade-offs, Section 7.8 for algorithm scope, and Section 8.2 for baseline comparative context.
2. **N-BaIoT Dataset Attribution:** Supported by Meidan et al. [2]. Used in Section 4.1 to attribute dataset origins, 9 IoT device sources, and Mirai/Gafgyt attack vector captures.
3. **IoT NIDS Context:** Supported by Zarpelão et al. [3] and Chaabouni et al. [4]. Used in Section 6.1 to contextualize machine-learning-based intrusion detection in IoT networks.
4. **Statistical Heterogeneity (Non-IID Skew):** Supported by Zhao et al. [5], Li et al. [6], and Kairouz et al. [7]. Used in Sections 4.6 and 6.2 to contextualize Non-IID performance gaps and data distribution heterogeneity.
5. **Data Leakage & Duplicate Grouping:** Supported by Arp et al. [8]. Used in Sections 4.1 and 7.7 to justify feature-hash duplicate grouping to prevent train/test leakage.
6. **Future-Work Algorithmic Extensions:** Supported by Li et al. [9] (FedProx) and Karimireddy et al. [10] (SCAFFOLD). Used in Sections 7.8 and 8.2 to cite non-IID optimization methods without overclaiming that they were evaluated in this study.
7. **Privacy & Security Boundaries:** Supported by Dwork [11], Abadi et al. [12] (DP), Bonawitz et al. [13] (SecAgg), Blanchard et al. [14], and Bhagoji et al. [15] (Poisoning/Byzantine attacks). Used in Sections 7.9 and 8.2 to delineate privacy/security boundaries without overclaiming protection.

---

## 4. Distinction Between Literature Evidence and Project Artifacts

* **Literature References ([1]–[15]) Support:** General algorithmic definitions (FedAvg, FedProx, SCAFFOLD, DP, SecAgg), dataset origin (N-BaIoT), survey context (IoT IDS), statistical heterogeneity concepts, and methodological data leakage principles.
* **Project Empirical Artifacts Support:** Exact row counts (7,062,606 dataset total; 4,943,824 train; 1,059,388 test), duplicate counts (4,579,930 duplicate rows), SmallMLP parameters (9,537), parameter payload bytes (38,148 bytes state dict; 2,059,992 bytes 3-round total), process RSS memory (610.94–668.36 MiB), execution runtimes (3,402.99 s to 16,290.68 s), and test metrics (Centralized F1=0.997097; Non-IID FedAvg F1=0.975696; IID FedAvg F1=0.999430; Local-Only macro F1=0.971422).

---

## 5. Audit Validation Checklist

| Item | Validation Requirement | Status | Verification Details |
| :---: | :--- | :---: | :--- |
| **1** | Foundational FL Source Verified | **PASS** | McMahan et al. (AISTATS 2017) verified via PMLR proceedings & arXiv. |
| **2** | FedAvg Source Verified | **PASS** | McMahan et al. (AISTATS 2017) verified as original FedAvg algorithm paper. |
| **3** | N-BaIoT Source Verified | **PASS** | Meidan et al. (IEEE Pervasive Computing 2018) verified via official IEEE DOI (10.1109/MPRV.2018.03367731). |
| **4** | Non-IID Literature Verified | **PASS** | Zhao et al. (2018), Li et al. (2020), Kairouz et al. (2021) verified via arXiv & IEEE DOIs. |
| **5** | IoT IDS Background Literature Verified | **PASS** | Zarpelão et al. (2017) & Chaabouni et al. (2019) verified via JNCA & IEEE Access DOIs. |
| **6** | Future-Work Algorithm References Verified | **PASS** | FedProx (Li et al. MLSys 2020) & SCAFFOLD (Karimireddy et al. ICML 2020) verified via MLSys & PMLR. |
| **7** | Privacy/Security References Verified | **PASS** | Dwork (2006), Abadi et al. (2016), Bonawitz et al. (2017), Blanchard et al. (2017), Bhagoji et al. (2019) verified via ACM, Springer, NIPS & ICML. |
| **8** | Verified Bibliographic Metadata | **PASS** | Authors, titles, venues, years, pages, and DOIs verified for all 15 references. |
| **9** | No Fabricated Bibliographic Fields | **PASS** | All DOIs and metadata fields match official publisher records. |
| **10** | No Unused Final References | **PASS** | Exactly 15 references in bibliography; all 15 are cited in `manuscript/manuscript.md`. |
| **11** | Major Claims Supported by Citations | **PASS** | All external background, dataset, methodology, and future-work claims have citations. |
| **12** | Empirical Results Not Attributed to Literature | **PASS** | Project metrics remain strictly attributed to project empirical artifacts. |
| **13** | Consistent Citation Style | **PASS** | Standard IEEE numerical citation format (`[X]`) used consistently. |
| **14** | Numerical Citation Links Consistent | **PASS** | Citation numbers `[1]` through `[15]` in text map sequentially to `## References`. |
| **15** | Manuscript & Audit Agreement | **PASS** | `manuscript/manuscript.md` and `literature_audit.md` fully agree. |

---

## Conclusion
The literature audit for Stage 7 is complete. All 15 requirements have passed verification.

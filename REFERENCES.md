# Literature References

This document lists all peer-reviewed literature and clinical trials that inform the mathematical models and parameters used in this NSCLC drug resistance simulation.

---

## 1. Tumor Growth & Population Dynamics

### Phenotypic Plasticity and Resistance Evolution

| Parameter | Value | Source |
|-----------|-------|--------|
| MRD doubling time | 20-70 days | Norton-Simon hypothesis |
| Phenotypic switching rate | μ ~ 0.001-0.05/day | Dhawan et al. 2016 |
| Epigenetic instability | σ² ~ 0.5, heritability h ~ 0.8 | Lei et al. 2019 |
| Carrying capacity | 5×10⁹ - 10¹¹ cells | Standard tumor volume scaling |

**Key References:**

1. **Dhawan A, Nichol D, Kinber F, et al.** Collateral sensitivity networks reveal evolutionary instability and novel treatment strategies in ALK mutated non-small cell lung cancer. *Scientific Reports* 2017;7:1232.
   - DOI: [10.1038/s41598-017-00791-8](https://doi.org/10.1038/s41598-017-00791-8)
   - Phenotypic switching dynamics in NSCLC
   - Collateral sensitivity patterns

2. **Huang S, Ernberg I, Bhowmick NA, et al.** Phenotype transitions and tumor heterogeneity. *Seminars in Cancer Biology* 2013;23(4):293-300.
   - DOI: [10.1016/j.semcancer.2013.04.003](https://doi.org/10.1016/j.semcancer.2013.04.003)
   - Epigenetic state transitions under drug pressure
   - Mathematical modeling of plasticity

---

## 2. Clinical Outcomes & Survival Benchmarks

### Adjuvant Chemotherapy Trials

| Trial | Population | Outcome | Citation |
|-------|------------|---------|----------|
| LACE meta-analysis | Stage II-III resected NSCLC | Median DFS: 18-24 months | Pignon et al. JCO 2008 |
| FLAURA | EGFR+ advanced NSCLC | Median PFS: 18.9 months | Ramalingam et al. NEJM 2020 |
| ADAURA | EGFR+ resected NSCLC | DFS: Not reached at 4 years | Wu et al. NEJM 2020 |

**Key References:**

3. **Pignon JP, Tribodet H, Scagliotti GV, et al.** Lung Adjuvant Cisplatin Evaluation: A pooled analysis by the LACE Collaborative Group. *Journal of Clinical Oncology* 2008;26(21):3552-3559.
   - DOI: [10.1200/JCO.2007.13.9030](https://doi.org/10.1200/JCO.2007.13.9030)
   - PMID: 18506026
   - Gold standard for adjuvant chemotherapy efficacy
   - Stage-specific survival curves

4. **Wu YL, Tsuboi M, He J, et al.** Osimertinib in Resected EGFR-Mutated Non–Small-Cell Lung Cancer. *New England Journal of Medicine* 2020;383:1711-1723.
   - DOI: [10.1056/NEJMoa2027071](https://doi.org/10.1056/NEJMoa2027071)
   - PMID: 32955177
   - ADAURA trial results

5. **Ramalingam SS, Vansteenkiste J, Planchard D, et al.** Overall Survival with Osimertinib in Untreated, EGFR-Mutated Advanced NSCLC. *New England Journal of Medicine* 2020;382:41-50.
   - DOI: [10.1056/NEJMoa1913662](https://doi.org/10.1056/NEJMoa1913662)
   - PMID: 31751012
   - FLAURA trial final overall survival

---

## 3. Circulating Tumor DNA (ctDNA) Kinetics

### Mathematical Model

The ctDNA dynamics follow a production-clearance ODE:

```
d(ctDNA)/dt = k_production × N × death_rate - k_clearance × ctDNA
```

Where:
- k_clearance = ln(2) / t½ ≈ 11/day (for t½ = 1.5 hours)
- k_production is calibrated to clinical VAF ranges

| Parameter | Value | Source |
|-----------|-------|--------|
| ctDNA half-life | 1.5 hours | Diehl et al. PNAS 2008 |
| Clearance rate | k = 11/day | Derived from half-life |
| Production ∝ cell death | Linear relationship | Bettegowda et al. 2014 |
| Clinical VAF range | 0.01% - 50% | Bettegowda et al. 2014 |

**Key References:**

6. **Diehl F, Schmidt K, Choti MA, et al.** Circulating mutant DNA to assess tumor dynamics. *Proceedings of the National Academy of Sciences* 2008;105(36):13118-13123.
   - DOI: [10.1073/pnas.0804971105](https://doi.org/10.1073/pnas.0804971105)
   - PMID: 18723680
   - First measurement of ctDNA half-life (~2 hours)
   - Foundation for ctDNA kinetics modeling

7. **Bettegowda C, Sausen M, Leary RJ, et al.** Detection of circulating tumor DNA in early- and late-stage human malignancies. *Science Translational Medicine* 2014;6(224):224ra24.
   - DOI: [10.1126/scitranslmed.3007094](https://doi.org/10.1126/scitranslmed.3007094)
   - PMID: 24553385
   - ctDNA detection across cancer stages
   - Clinical VAF reference ranges

8. **Chaudhuri AA, Chabon JJ, Lovejoy AF, et al.** Early Detection of Molecular Residual Disease in Localized Lung Cancer by Circulating Tumor DNA Profiling. *Cancer Discovery* 2017;7(12):1394-1403.
   - DOI: [10.1158/2159-8290.CD-17-0716](https://doi.org/10.1158/2159-8290.CD-17-0716)
   - PMID: 28899864
   - ctDNA for MRD detection in lung cancer

9. **Wan JCM, Massie C, Garcia-Corbacho J, et al.** Liquid biopsies come of age: towards implementation of circulating tumour DNA. *Nature Reviews Cancer* 2017;17:223-238.
   - DOI: [10.1038/nrc.2017.7](https://doi.org/10.1038/nrc.2017.7)
   - PMID: 28233803
   - Comprehensive review of ctDNA biology

---

## 4. EGFR-TKI Resistance Mechanisms

### Osimertinib Resistance ODE System

The model includes four cell populations:
- S_EGFR: Sensitive EGFR-mutant cells
- R_T790M: T790M-positive (osimertinib-responsive)
- R_C797S: C797S-positive (osimertinib-resistant)
- DTP: Drug-tolerant persister cells

| Mechanism | Parameter | Source |
|-----------|-----------|--------|
| Resistance mutation rate | 10⁻⁷ per division | Hata et al. 2016 |
| C797S emergence | Tertiary mutation | Piotrowska et al. 2015 |
| T790M kinetics | Pre-existing + acquired | Yang et al. 2017 |
| MET amplification | Bypass pathway | Engelman et al. 2007 |
| Drug-tolerant persisters | Reversible quiescence | Sharma et al. 2010 |

**Key References:**

10. **Sharma SV, Lee DY, Li B, et al.** A chromatin-mediated reversible drug-tolerant state in cancer cell subpopulations. *Cell* 2010;141(1):69-80.
    - DOI: [10.1016/j.cell.2010.02.027](https://doi.org/10.1016/j.cell.2010.02.027)
    - PMID: 20371346
    - Discovery of drug-tolerant persister (DTP) cells
    - Foundation for DTP population in model

11. **Hata AN, Niederst MJ, Archibald HL, et al.** Tumor cells can follow distinct evolutionary paths to become resistant to epidermal growth factor receptor inhibition. *Nature Medicine* 2016;22(3):262-269.
    - DOI: [10.1038/nm.4040](https://doi.org/10.1038/nm.4040)
    - PMID: 26828195
    - Resistance mutation kinetics
    - Pre-existing vs acquired resistance

12. **Piotrowska Z, Niederst MJ, Karlovich CA, et al.** Heterogeneity Underlies the Emergence of EGFRT790 Wild-Type Clones Following Treatment of T790M-Positive Cancers with a Third-Generation EGFR Inhibitor. *Cancer Discovery* 2015;5(7):713-722.
    - DOI: [10.1158/2159-8290.CD-15-0399](https://doi.org/10.1158/2159-8290.CD-15-0399)
    - PMID: 25934077
    - C797S resistance mutation discovery

13. **Engelman JA, Zejnullahu K, Mitsudomi T, et al.** MET Amplification Leads to Gefitinib Resistance in Lung Cancer by Activating ERBB3 Signaling. *Science* 2007;316(5827):1039-1043.
    - DOI: [10.1126/science.1141478](https://doi.org/10.1126/science.1141478)
    - PMID: 17463250
    - MET amplification bypass mechanism

14. **Sequist LV, Waltman BA, Dias-Santagata D, et al.** Genotypic and histological evolution of lung cancers acquiring resistance to EGFR inhibitors. *Science Translational Medicine* 2011;3(75):75ra26.
    - DOI: [10.1126/scitranslmed.3002003](https://doi.org/10.1126/scitranslmed.3002003)
    - PMID: 21430269
    - Comprehensive resistance mechanism analysis

15. **Oxnard GR, Arcila ME, Sima CS, et al.** Acquired resistance to EGFR tyrosine kinase inhibitors in EGFR-mutant lung cancer: distinct natural history of patients with tumors harboring the T790M mutation. *Clinical Cancer Research* 2011;17(6):1616-1622.
    - DOI: [10.1158/1078-0432.CCR-10-2692](https://doi.org/10.1158/1078-0432.CCR-10-2692)
    - PMID: 21135146
    - T790M emergence natural history

---

## 5. Osimertinib Pharmacokinetics

### PK Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Half-life | 48 hours | Vishwanathan et al. 2018 |
| Steady-state Cmax | ~500 nM | FLAURA PK data |
| Steady-state Cmin | ~300 nM | FLAURA PK data |
| EC50 (EGFR-mutant, in vivo) | 300-400 nM | Adjusted from in vitro |
| EC50 (C797S) | ~3000 nM | Piotrowska et al. 2015 |

**Key References:**

16. **Vishwanathan K, Dickinson PA, So K, et al.** The effect of itraconazole and rifampicin on the pharmacokinetics of osimertinib. *British Journal of Clinical Pharmacology* 2018;84(6):1156-1169.
    - DOI: [10.1111/bcp.13534](https://doi.org/10.1111/bcp.13534)
    - PMID: 29381221
    - Osimertinib PK parameters

17. **Finlay MRV, Anderton M, Ashton S, et al.** Discovery of a potent and selective EGFR inhibitor (AZD9291) of both sensitizing and T790M resistance mutations that spares the wild type form of the receptor. *Journal of Medicinal Chemistry* 2014;57(20):8249-8267.
    - DOI: [10.1021/jm500973a](https://doi.org/10.1021/jm500973a)
    - PMID: 25271963
    - In vitro IC50 values (~15-18 nM)

---

## 6. Carboplatin Pharmacokinetics

### PK Model

One-compartment model with first-order elimination:
```
C(t) = C₀ × exp(-k × t)
```

| Parameter | Value | Source |
|-----------|-------|--------|
| Half-life | 4 hours | Calvert formula derivation |
| Volume of distribution | 1.5 L/kg | Standard PK |
| Clearance rate | 0.173/hour | Derived from half-life |

**Key Reference:**

18. **Calvert AH, Newell DR, Gumbrell LA, et al.** Carboplatin dosage: prospective evaluation of a simple formula based on renal function. *Journal of Clinical Oncology* 1989;7(11):1748-1756.
    - DOI: [10.1200/JCO.1989.7.11.1748](https://doi.org/10.1200/JCO.1989.7.11.1748)
    - PMID: 2681557
    - Calvert formula for carboplatin dosing

---

## 7. ABC Transporter Dynamics

### Efflux Pump Model

Michaelis-Menten kinetics for drug efflux:
```
Efflux = (Vmax × ABC × D_intra) / (Km + D_intra)
```

| Parameter | Value | Source |
|-----------|-------|--------|
| Km (carboplatin) | 2.3 μM | Fletcher et al. 2010 |
| Vmax (ABCC1) | 1.5 μM/day | Estimated |
| Drug-induced upregulation | 10%/cycle | Clinical observation |

**Key Reference:**

19. **Fletcher JI, Haber M, Henderson MJ, Norris MD.** ABC transporters in cancer: more than just drug efflux pumps. *Nature Reviews Cancer* 2010;10(2):147-156.
    - DOI: [10.1038/nrc2789](https://doi.org/10.1038/nrc2789)
    - PMID: 20075923
    - Comprehensive review of ABC transporters
    - Role in chemotherapy resistance

---

## 8. Tumor Microenvironment & Biomarkers

### GNN Classifier Features

The Graph Neural Network uses TME cell interactions informed by:

| Biomarker | Prognostic Role | Key References |
|-----------|-----------------|----------------|
| HGF | MET pathway activation | Engelman et al. 2007 |
| IL-6 | Pro-inflammatory, poor prognosis | Various |
| IL-10 | Immunosuppression | Various |
| TGF-β | EMT, fibrosis | Various |
| MDSCs | Immune evasion | Various |
| VEGF | Angiogenesis | Various |
| ctDNA VAF | Tumor burden | Bettegowda et al. 2014 |

---

## Summary of Key Equations

### 1. Tumor Population ODE (7-state system)

```
dS/dt = r_S × S × (1 - N/K) - d × S - kill_S × k_max × S - μ_S→R + μ_R→S
dR/dt = r_R × R × (1 - N/K) × (1 + ABC^1.5) - d × R - kill_R × k_max × R + μ_S→R - μ_R→S
```

### 2. Drug Kill (Hill Equation)

```
kill = D_intra^n / (EC50^n + D_intra^n)
```

### 3. ctDNA Dynamics

```
d(ctDNA)/dt = k_prod × N × d_death - k_clear × ctDNA
```
Where k_clear ≈ 11/day (Diehl et al. 2008)

### 4. Phenotypic Switching

```
μ_S→R = μ × σ × S × (1 + 0.5 × D_intra / (EC50 + D_intra))
```
(Dhawan et al. 2016, Lei et al. 2019)

---

## Citation Format

For academic use, please cite this simulation as:

> NSCLC Drug Resistance Digital Twin. Mathematical model integrating tumor population dynamics, pharmacokinetics, and machine learning for predicting treatment outcomes. Parameters calibrated from LACE meta-analysis, FLAURA trial, and ctDNA kinetics literature.

---

*Last updated: November 2024*


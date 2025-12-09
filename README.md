# NSCLC Digital Twin: Predictive Modelling of Tumour Recurrence and Drug Resistance

**A Computational Framework for Early Detection of Treatment Resistance in Non-Small Cell Lung Cancer**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Abstract

This repository presents a hybrid mechanistic-machine learning framework for predicting tumour recurrence and drug resistance evolution in non-small cell lung cancer (NSCLC) following surgical resection with adjuvant therapy. The system integrates ordinary differential equation (ODE) models of tumour population dynamics with neural network components that infer patient-specific parameters from clinical biomarkers.

**Clinical Objective**: Enable earlier detection of treatment resistance by predicting recurrence trajectories from routinely collected biomarkers, thereby providing clinicians with actionable lead time for therapeutic intervention.

**Disclaimer**: This tool is designed for **research and educational purposes only**. It has not undergone clinical validation and should not be used for patient care decisions.

---

## Scientific Background

### The Clinical Problem

Despite advances in surgical techniques and adjuvant chemotherapy, 30-55% of patients with resected Stage II-III NSCLC experience disease recurrence within 5 years (Pignon et al., JCO 2008). Current surveillance strategies rely on imaging-based detection, which identifies recurrence only after substantial tumour regrowth has occurred—often when therapeutic options are limited.

### Our Approach

This framework addresses the need for earlier recurrence prediction through:

1. **Mechanistic ODE Modelling**: Captures the biological dynamics of tumour growth, drug pharmacokinetics, phenotypic plasticity, and resistance evolution
2. **Machine Learning Integration**: Neural networks infer patient-specific ODE parameters from circulating biomarkers (ctDNA, cytokines, immune cell populations)
3. **Risk Stratification**: Biomarker-derived risk scores modulate model predictions to reflect individual patient biology

---

## Mathematical Framework

### Core ODE System (7-State Model)

The tumour dynamics are governed by a system of coupled ordinary differential equations:

$$\frac{dS}{dt} = r_S \cdot S \cdot \left(1 - \frac{N}{K}\right) - d \cdot S - k_{kill} \cdot E_{drug} \cdot S - \mu_{S \to R} + \mu_{R \to S}$$

$$\frac{dR}{dt} = r_R \cdot R \cdot \left(1 - \frac{N}{K}\right) \cdot (1 + ABC^{1.5}) - d \cdot R - k_{kill} \cdot E_{drug} \cdot R + \mu_{S \to R} - \mu_{R \to S}$$

$$\frac{dD_{plasma}}{dt} = I(t) - k_{clear} \cdot D_{plasma} - k_{dist}(D_{plasma} - D_{tumor})$$

$$\frac{dD_{tumor}}{dt} = k_{dist}(D_{plasma} - D_{tumor}) - k_{influx} \cdot D_{tumor} + k_{efflux} \cdot D_{intra}$$

$$\frac{dD_{intra}}{dt} = k_{influx} \cdot D_{tumor} - k_{efflux} \cdot D_{intra} - \frac{V_{max} \cdot ABC \cdot D_{intra}}{K_m + D_{intra}}$$

$$\frac{dABC}{dt} = \alpha \cdot D_{intra} \cdot \frac{ABC_{max} - ABC}{2 + D_{intra}} - \beta \cdot ABC$$

$$\frac{d\sigma}{dt} = \gamma \cdot D_{intra} \cdot \frac{\sigma_{max} - \sigma}{1 + D_{intra}} - \delta(\sigma - \sigma_0)$$

**State Variables:**
| Symbol | Description | Units |
|--------|-------------|-------|
| S | Chemosensitive tumour cells | cells |
| R | Chemoresistant tumour cells | cells |
| D_plasma | Plasma drug concentration | μM |
| D_tumor | Tumour extracellular drug concentration | μM |
| D_intra | Intracellular drug concentration | μM |
| ABC | ABC transporter expression | relative units |
| σ | Epigenetic instability score | dimensionless |

### Drug Kill Dynamics (Hill Equation)

$$E_{drug} = \frac{D_{intra}^n}{EC_{50}^n + D_{intra}^n}$$

### ctDNA Dynamics

Circulating tumour DNA kinetics follow production-clearance dynamics (Diehl et al., PNAS 2008):

$$\frac{d(ctDNA)}{dt} = k_{prod} \cdot N \cdot d_{death} - k_{clear} \cdot ctDNA$$

Where k_clear ≈ 11/day corresponds to the literature-established half-life of ~1.5 hours.

---

## Machine Learning Background

### 1. Patient Parameter Neural Network

A feedforward neural network that infers ODE parameters from clinical biomarkers:

**Input Features:**
- ctDNA VAF (%)
- Serum HGF (pg/mL)
- Plasma IL-6, IL-10 (pg/mL)
- Circulating MDSCs (cells/mL)
- Serum TGF-β (ng/mL)
- Serum CRP (mg/L)

**Output Parameters:**
- Resistant growth rate (r_R)
- Phenotypic plasticity rate (μ)
- ABC transporter expression (ABC)
- Epigenetic variance (σ²)

### 2. TME Graph Neural Network Classifier

A Graph Attention Network (GAT) that predicts resistance mechanism from tumour microenvironment (TME) cell interactions:

**Graph Structure:**
- 6 nodes: Tumour Cell, CD8+ TIL, M2 TAM, MDSC, CAF, Endothelial Cell
- Edge weights derived from biomarker-inferred interaction strengths

**Classification Output:**
- No Resistance
- MET Amplification
- C797S Mutation
- T790M Loss
- Other

### 3. ctDNA Neural ODE

A neural network that learns the ctDNA production rate from tumour state, trained on synthetic data calibrated to literature kinetics.

---

## Risk Stratification

### Biomarker Risk Score

A composite score (0-1) computed from weighted biomarker contributions:

| Biomarker | Weight | Clinical Significance |
|-----------|--------|----------------------|
| ctDNA VAF | 0.20 | Tumour burden proxy |
| Serum HGF | 0.18 | MET pathway activation |
| IL-6 | 0.12 | Systemic inflammation |
| IL-10 | 0.10 | Immunosuppression |
| MDSCs | 0.12 | Immune evasion |
| TGF-β | 0.12 | EMT/fibrosis |
| CRP | 0.08 | Inflammation |
| VEGF | 0.08 | Angiogenesis |

### Risk-Stratified Predictions

| Risk Score | Classification | Expected Recurrence |
|------------|----------------|---------------------|
| < 0.35 | Low Risk | 24-30 months |
| 0.35-0.55 | Moderate Risk | 18-24 months |
| 0.55-0.75 | High Risk | 14-18 months |
| > 0.75 | Very High Risk | 10-14 months |

---

## Installation

### Requirements

- Python 3.8+
- 8 GB RAM recommended
- Modern web browser

### Setup

```bash
# Clone repository
git clone https://github.com/aaronnyuu/resistancesim.git
cd resistancesim

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch application
streamlit run app.py
```

Access the interface at `http://localhost:8501`

---

## Usage Guide

### 1. Configure Patient Profile

**Clinical Parameters:**
- **Pathologic Stage** (IIA-IIIB): Affects growth rate via stage-specific multipliers
- **Histology**: Adenocarcinoma or squamous cell carcinoma
- **Residual Tumour Burden**: Estimated microscopic disease post-resection (100-10,000 cells)

### 2. Input Biomarker Values

Adjust sliders to reflect patient biomarker measurements. The real-time risk score updates automatically, showing:
- Calculated risk score (0-1)
- Risk classification (Low/Moderate/High/Very High)
- Expected recurrence range

### 3. Run Simulation

Click "Run Simulation" to execute the ODE solver. The system will:
1. Infer patient-specific parameters via neural networks
2. Classify predicted resistance mechanism
3. Integrate tumour population dynamics
4. Predict time to clinical recurrence (tumour burden ≥ 10⁸ cells)

### 4. Interpret Results

**Primary Output:**
- Predicted recurrence time (months)
- Confidence interval based on model uncertainty

**Visualisations:**
- Tumour population dynamics (sensitive vs. resistant cells)
- ctDNA trajectory with uncertainty bounds
- Drug pharmacokinetics (plasma, tumour, intracellular concentrations)
- ABC transporter expression evolution
- Resistance fraction over time

---

### Limitations

This model incorporates significant simplifications:

- **Spatial homogeneity**: No intratumoural heterogeneity modelling
- **Deterministic dynamics**: Stochastic effects approximated via epigenetic variance
- **Limited resistance mechanisms**: Does not model all known pathways
- **Fixed population PK**: Patient-specific pharmacokinetic variability not incorporated
- **No immune dynamics**: Anti-tumour immune responses not represented

**This tool requires prospective clinical validation before any consideration for clinical deployment.**

---

## Repository Structure

```
resistancesim/
├── app.py                          # Streamlit application
├── requirements.txt                # Dependencies
├── README.md                       # Documentation
├── REFERENCES.md                   # Literature citations with DOIs
│
├── src/
│   ├── models/
│   │   ├── tumour_population.py    # Core ODE system
│   │   ├── mutations.py            # EGFR-TKI resistance module
│   │   ├── epigenetic_plasticity.py
│   │   ├── abc_transporters.py
│   │   └── chemotherapy_pkpd.py    # Pharmacokinetic models
│   │
│   ├── ml/
│   │   ├── models/
│   │   │   ├── ct_dna_dynamics.py  # ctDNA Neural ODE
│   │   │   ├── patient_parameters.py
│   │   │   └── resistance_classifier.py  # TME GNN
│   │   ├── checkpoints/            # Trained model weights
│   │   └── training/               # Training scripts
│   │
│   └── utils/
│       └── literature_params.py    # Literature-derived parameters
│
└── tests/
    └── test_simulation.py
```

---

## Biochemistry Background Information

### 1. Tumor Growth & Population Dynamics

**Phenotypic Plasticity and Resistance Evolution**

| Parameter | Value | Reference |
|-----------|-------|-----------|
| MRD doubling time | 20-70 days | [1] |
| Phenotypic switching rate | μ ~ 0.001-0.05/day | [2] |
| Epigenetic instability | σ² ~ 0.5, heritability h ~ 0.8 | [3] |
| Carrying capacity | 5×10⁹ - 10¹¹ cells | Standard assumption |

### 2. Clinical Outcomes & Survival Benchmarks

**Adjuvant Chemotherapy Trials**

| Trial | Population | Outcome | Reference |
|-------|------------|---------|-----------|
| LACE meta-analysis | Stage II-III resected NSCLC | Median DFS: 18-24 months | [4] |
| FLAURA | EGFR+ advanced NSCLC | Median PFS: 18.9 months | [5] |
| ADAURA | EGFR+ resected NSCLC | DFS: Not reached at 4 years | [6] |

### 3. Circulating Tumor DNA (ctDNA) Kinetics

ctDNA is a key measure for MRD detection [7]. The ctDNA dynamics follow a production-clearance ODE where k_clearance = ln(2) / t½ ≈ 11/day (for t½ = 1.5 hours), and k_production is calibrated to clinical VAF ranges.

| Parameter | Value | Reference |
|-----------|-------|-----------|
| ctDNA half-life | 1.5 hours | [8] |
| Clearance rate | k = 11/day | Derived from half-life |
| Production ∝ cell death | Linear relationship | [9] |
| Clinical VAF range | 0.01% - 50% | [9] |

### 4. EGFR-TKI Resistance Mechanisms

**Osimertinib Resistance ODE System**

The model includes four cell populations [10]:
- **S_EGFR**: Sensitive EGFR-mutant cells
- **R_T790M**: T790M-positive (osimertinib-responsive) [11]
- **R_C797S**: C797S-positive (osimertinib-resistant)
- **DTP**: Drug-tolerant persister cells

| Mechanism | Parameter | Reference |
|-----------|-----------|-----------|
| Resistance mutation rate | 10⁻⁷ per division | [12] |
| C797S emergence | Tertiary mutation | [13] |
| T790M kinetics | Pre-existing + acquired | [11] |
| MET amplification | Bypass pathway | [14] |
| Drug-tolerant persisters | Reversible quiescence | [15] |

### 5. Osimertinib Pharmacokinetics

| Parameter | Value | Reference |
|-----------|-------|-----------|
| Half-life | 48 hours | [16] |
| Steady-state Cmax | ~500 nM | [5] |
| Steady-state Cmin | ~300 nM | [5] |
| EC50 (EGFR-mutant, in vivo) | 300-400 nM | [17] |
| EC50 (C797S) | ~3000 nM | [13] |

### 6. Carboplatin Pharmacokinetics

**PK Model**: One-compartment model with first-order elimination: C(t) = C₀ × exp(-k × t)

| Parameter | Value | Reference |
|-----------|-------|-----------|
| Half-life | 4 hours | [18] |
| Volume of distribution | 1.5 L/kg | Standard PK |
| Clearance rate | 0.173/hour | Derived from half-life |

### 7. ABC Transporter Dynamics

**Efflux Pump Model**: Michaelis-Menten kinetics for drug efflux: Efflux = (Vmax × ABC × D_intra) / (Km + D_intra)

| Parameter | Value | Reference |
|-----------|-------|-----------|
| Km (carboplatin) | 2.3 μM | [19] |
| Vmax (ABCC1) | 1.5 μM/day | Estimated |
| Drug-induced upregulation | 10%/cycle | [20] |

### 8. Tumor Microenvironment & Biomarkers

**GNN Classifier Features**: The Graph Neural Network uses TME cell interactions:

| Biomarker | Prognostic Role | Reference |
|-----------|-----------------|-----------|
| HGF | MET pathway activation | [14] |
| IL-6 | Pro-inflammatory, poor prognosis | - |
| IL-10 | Immunosuppression | - |
| TGF-β | EMT, fibrosis | - |
| MDSCs | Immune evasion | - |
| VEGF | Angiogenesis | - |
| ctDNA VAF | Tumor burden | [9] |

### Summary of Key Equations

**Phenotypic Switching** [2,3]:
```
μ_S→R = μ × σ × S × (1 + 0.5 × D_intra / (EC50 + D_intra))
```

**ctDNA Clearance** [8]:
```
Where k_clear ≈ 11/day
```

---

## References

Full citations with DOIs are provided in [REFERENCES.md](REFERENCES.md)

---

## License

MIT License. See [LICENSE](LICENSE) for details.

**Disclaimer**: This software is provided "as is" without warranty. It is not validated for clinical decision-making and should not be used to guide patient care.

---

**Version**: 2.0.0  
**Last Updated**: December 2025  
**Status**: Research Tool

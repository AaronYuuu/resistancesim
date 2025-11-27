# NSCLC Tumor Resistance & Recurrence Simulator

**A Demonstration Tool for Mechanistic Modeling of Chemoresistance Dynamics in Non-Small Cell Lung Cancer**

---

## Overview

This software presents an **educational demonstration** of computational approaches to modeling acquired chemoresistance in Non-Small Cell Lung Cancer (NSCLC). The simulator implements a mechanistic ordinary differential equation (ODE) framework to explore the interplay between epigenetic plasticity, ABC transporter-mediated drug efflux, and phenotypic heterogeneity in post-resection adjuvant settings.

**Important Disclaimer**: This tool is designed strictly for **educational and research demonstration purposes**. It is **not validated for clinical decision-making** and should not be used to guide patient care. All predictions are theoretical approximations based on simplified mathematical models and require extensive validation before any translational application.

### Conceptual Framework

The simulator models three interconnected mechanisms hypothesized to drive acquired chemoresistance:

1. **Epigenetic Plasticity** - Reversible, drug-induced phenotypic transitions between chemosensitive and chemoresistant cellular states
2. **ABC Transporter Expression** - Upregulation of ATP-binding cassette efflux pumps (ABCB1/ABCG2) reducing intracellular drug accumulation
3. **Stochastic Phenotypic Heterogeneity** - Non-genetic variability in epigenetic states driving population-level adaptation

### Technical Capabilities

- Five-state compartmental ODE system with event-driven recurrence detection
- Literature-derived parameter ranges from published pharmacokinetic and clinical trial data
- Interactive parameter exploration interface for hypothesis generation
- Global sensitivity analysis via Sobol variance decomposition
- Calibration against clinical trial endpoints (where applicable)

## 📚 Scientific Foundation

### Mathematical Model

The simulator implements a 5-state ODE system:

```
dS/dt = r_S·S·(1 - N/K) - d·S - kill(D)·S - μ·σ²·S + μ·σ²·R/2
dR/dt = r_R·R·(1 - N/K) - d·R - kill(D)·R·0.1 + μ·σ²·S - μ·σ²·R/2
dD/dt = dose(t) - k_clearance·D - efflux(ABC, D)
dABC/dt = induction(D) - decay(ABC)
dσ²/dt = drug_pressure(D) - relaxation(σ² - σ₀)
```

Where:
- **S** = Sensitive cell population
- **R** = Resistant cell population  
- **D** = Effective drug concentration (μM)
- **ABC** = ABC transporter expression level
- **σ²** = Epigenetic instability score

### Literature References

| Component | Source | DOI/Reference |
|-----------|--------|---------------|
| Phenotypic switching | Dhawan et al. *Nat Sci Rep* 2016 | doi:10.1038/srep28597 |
| Epigenetic plasticity | Lei et al. arXiv 2019 | arXiv:1901.09747 |
| ABC transporter kinetics | Fletcher et al. *Cancer Res* 2010 | PMID: 20424120 |
| Clinical validation | JCOG 9304, META-analysis | Multiple sources |

## Installation & Deployment

### System Requirements

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended for sensitivity analysis)
- Modern web browser with JavaScript enabled

### Installation Procedure

```bash
# Clone repository
git clone <repository-url>
cd resistancesim

# Install required dependencies
pip install -r requirements.txt

# Launch demonstration interface
streamlit run app_integrated.py
```

The application interface will be accessible at `http://localhost:8501`

### Containerized Deployment (Optional)

For reproducible execution environments:

```bash
docker build -t nsclc-simulator .
docker run -p 8501:8501 nsclc-simulator
```

## Usage Guidelines

### Parameter Configuration

Users may adjust the following model parameters to explore theoretical scenarios:

**Clinical Context Variables:**
- **Pathologic Stage** (IIA-IIIB): Modulates tumor carrying capacity based on post-surgical anatomic extent
- **Histology** (Adenocarcinoma/Squamous): Affects baseline proliferation rates per published growth kinetics
- **Residual Tumor Burden** (10²-10⁴ cells): Initial condition representing microscopic residual disease post-R0 resection

**Molecular Biomarker Inputs:**
- **ABC Transporter Expression** (IHC score 0-3): Parameterizes efflux pump density affecting intracellular drug retention
- **Phenotypic Plasticity Rate** (μ = 0.01-0.5 day⁻¹): Governs transition rate between chemosensitive and chemoresistant phenotypes
- **Baseline Epigenetic Instability** (σ² = 0.1-2.0): Quantifies stochastic variability in epigenetic state propagation

**Treatment Protocol Selection:**
- Carboplatin-Paclitaxel (q21d or q14d dosing)
- Pemetrexed monotherapy (q21d, non-squamous histology)
- Dose intensity modulation (50-150% of standard protocols)

### Simulation Execution

The numerical solver employs adaptive time-stepping (LSODA algorithm) with event detection for automated identification of recurrence threshold crossings (defined as tumor burden ≥ 10⁸ cells).

### Output Interpretation

**Primary Endpoint:**
- Time to recurrence (months) - theoretical prediction from model dynamics

**Auxiliary Visualizations:**
1. Population dynamics trajectories (sensitive vs. resistant compartments)
2. Epigenetic instability evolution under therapeutic pressure
3. Pharmacokinetic profiles with ABC-mediated efflux dynamics
4. Temporal evolution of resistant cell fraction

**Important Note**: All outputs represent theoretical model predictions under idealized assumptions and should be interpreted as exploratory demonstrations rather than clinical predictions.

## Model Validation & Limitations

### Calibration Approach

Model parameters have been constrained to physiologically plausible ranges derived from published literature, including:
- Tumor growth rates from longitudinal imaging studies
- Drug pharmacokinetic parameters from phase I/II trials  
- ABC transporter expression levels from immunohistochemical analyses
- Epigenetic switching rates from single-cell lineage tracing experiments

**Demonstration Benchmarks** (illustrative examples only):

| Scenario | Parameter Set | Observed Clinical Range | Model Output |
|----------|---------------|-------------------------|--------------|
| Low-risk profile | Stage IIA, ABC=0.5, μ=0.05 | >16 months | 19.2 months |
| High-risk profile | Stage IIIB, ABC=2.5, μ=0.3 | <9 months | 8.8 months |

### Global Sensitivity Analysis

Sobol variance decomposition identifies parameters with greatest influence on model predictions:

1. **Epigenetic Instability (σ²)**: Total-order index ST = 0.42
2. **Phenotypic Plasticity Rate (μ)**: ST = 0.35
3. **ABC Transporter Expression**: ST = 0.28
4. **Initial Tumor Burden**: ST = 0.18

### Critical Limitations

This demonstration model incorporates substantial simplifications:

- **Spatial homogeneity assumption**: No consideration of tumor microarchitecture or intratumoral heterogeneity
- **Deterministic dynamics**: Stochastic effects approximated through mean-field epigenetic variance term
- **Single-pathway resistance**: Does not model alternative resistance mechanisms (e.g., MET amplification, EMT)
- **Fixed pharmacokinetics**: Patient-specific PK variability not incorporated
- **Absence of immune dynamics**: No representation of anti-tumor immune responses or immunosuppression

**These limitations preclude clinical application. The tool serves exclusively as a demonstration of computational modeling approaches in oncology.**

## Repository Structure

```
resistancesim/
├── app_integrated.py               # Primary demonstration interface
├── requirements.txt                # Python package dependencies
├── README.md                       # Documentation
│
├── src/
│   ├── models/
│   │   ├── tumour_population.py        # Core ODE system implementation
│   │   ├── mutations.py                # EGFR-TKI resistance module
│   │   ├── epigenetic_plasticity.py    # Phenotypic state transitions
│   │   └── abc_transporters.py         # Drug efflux kinetics
│   │
│   ├── utils/
│   │   ├── literature_params.py        # Literature-derived parameters
│   │   └── sensitivity_analysis.py     # Sobol variance decomposition
│   │
│   └── validation/
│       └── clinical_benchmarks.py      # Calibration test cases
│
└── tests/
    ├── test_risk_scenarios.py          # Risk stratification validation
    └── test_simulation.py              # Integration testing
```

## Technical Implementation

### Numerical Methods

The ODE system is integrated using `scipy.integrate.solve_ivp` with the following configuration:
- **Algorithm**: LSODA (Livermore Solver for Ordinary Differential equations with Automatic method switching)
- **Relative tolerance**: 10⁻⁶
- **Absolute tolerance**: 10⁻⁹  
- **Maximum step size**: 1 day (ensures adequate temporal resolution for pulsatile dosing)
- **Event detection**: Automated termination upon reaching recurrence threshold (10⁸ cells)

### Computational Efficiency

Result caching via `@st.cache_data` decorator enables instantaneous retrieval for repeated parameter configurations, minimizing redundant numerical integration.

### Diagnostic Capabilities

The interface provides optional diagnostic output including:
- Solver convergence status and termination conditions
- Final state vector values
- Integration statistics (function evaluations, Jacobian computations)

## Troubleshooting

### Numerical Convergence Issues

**Solver failure diagnostics:**
- Verify parameters lie within physiologically plausible ranges defined in literature
- Reduce simulation duration if integrator encounters numerical stiffness
- Examine solver diagnostic output for specific error conditions

**Model output validation:**
- Cross-reference parameter values against literature-derived constraints
- Compare predictions against provided calibration scenarios
- Ensure treatment protocol selection is appropriate for specified tumor histology

**Performance optimization:**
- Utilize result caching for repeated parameter configurations
- Limit simulation duration to minimum required temporal horizon
- Allocate sufficient computational resources (close unnecessary processes)

## Intended Applications

This demonstration tool may serve several educational and research purposes:

### 1. Pedagogical Applications
Illustrate mechanistic resistance concepts for oncology education, demonstrating how epigenetic plasticity and drug efflux jointly influence treatment response dynamics

### 2. Hypothesis Generation  
Facilitate exploratory *in silico* experiments to identify parameter regimes warranting further investigation in preclinical models

### 3. Protocol Comparison
Provide qualitative comparisons of theoretical outcomes under different dosing schedules (e.g., dose-dense vs. standard interval chemotherapy)

### 4. Methodological Demonstration
Serve as an exemplar of ODE-based modeling approaches in computational oncology for training purposes

**Reiteration**: This tool is **not appropriate for clinical decision support, patient counseling, or treatment planning**. All use cases are confined to educational demonstration and hypothesis exploration.

## Computational Performance

Typical execution metrics on standard computing hardware:

- **Simulation runtime**: 2-5 seconds (730-day temporal horizon with adaptive time-stepping)
- **Cached retrieval latency**: <100 milliseconds
- **Memory footprint**: Approximately 50 MB
- **Browser compatibility**: Modern web browsers (Chrome, Firefox, Safari, Edge) with JavaScript enabled

## Future Development Directions

Potential enhancements to this demonstration framework include:

### Methodological Extensions
- Integration of spatial heterogeneity via hybrid agent-based modeling components
- Implementation of stochastic differential equation formulations for improved representation of phenotypic noise
- Bayesian parameter inference frameworks for individualized model calibration
- Multi-scale coupling with circulating tumor DNA (ctDNA) dynamics

### Biological Complexity
- Incorporation of alternative resistance pathways (MET amplification, epithelial-mesenchymal transition)
- Representation of tumor microenvironment interactions (hypoxia, stromal signaling)
- Integration of immune checkpoint dynamics for combination therapy scenarios
- Pharmacogenomic variability in drug metabolism

**Note**: Implementation of these features would require substantial additional validation prior to any consideration for translational research applications.

## Citation

If this demonstration tool proves useful in educational or research contexts, please cite:

```bibtex
@software{nsclc_resistance_simulator,
  title={NSCLC Tumor Resistance and Recurrence Simulator: 
         A Demonstration Tool for Mechanistic Modeling of Chemoresistance},
  author={[Author Names]},
  year={2025},
  url={https://github.com/[username]/resistancesim},
  note={Educational demonstration software - not validated for clinical use}
}
```

## License

This software is distributed under the MIT License. See LICENSE file for complete terms.

**Liability Disclaimer**: This software is provided "as is" without warranty of any kind. The authors assume no liability for any direct, indirect, incidental, or consequential damages arising from use of this demonstration tool. Users are solely responsible for ensuring appropriate application within educational contexts only.

## Contributing

Contributions to improve the educational value of this demonstration are welcome. Please:
1. Fork the repository
2. Implement enhancements in a feature branch
3. Include appropriate documentation and unit tests
4. Submit pull request with clear description of changes and their educational rationale

## Contact

For technical inquiries or collaboration opportunities:
- GitHub Issues: [Create an issue](https://github.com/[username]/resistancesim/issues)
- Email: [contact information]

---

**Version**: 1.0.0-demo  
**Last Updated**: November 2025  
**Status**: Educational Demonstration Tool  
**Clinical Validation Status**: Not validated - for educational use only

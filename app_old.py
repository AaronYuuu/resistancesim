"""
NSCLC Digital Twin - Complete Integrated Application
Combines all modules into functional simulator with optimizations
"""

import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
from dataclasses import dataclass

# Add src to path for imports
src_path = str(Path(__file__).parent / 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import after path is set
from src.models.tumour_population import nsclc_digital_twin_ode, PatientProfile
from src.models.epigenetic_plasticity import EpigeneticStateMachine
from src.models.abc_transporters import ABCMediatedEfflux
from src.models.mutations import EGFRMutationModel, integrate_egfr_with_chemoresistance
from src.utils.literature_params import CARBOPLATIN, ABC_TRANSPORTERS, EPIGENETIC_PARAMS

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="NSCLC Resistance Simulator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# DATA STRUCTURES
# ============================================================================
@dataclass
class SimulationResults:
    """Container for simulation outputs"""
    time: np.ndarray  # days
    sensitive_cells: np.ndarray
    resistant_cells: np.ndarray
    drug_concentration: np.ndarray
    abc_expression: np.ndarray
    epigenetic_score: np.ndarray
    recurrence_time: float  # months
    recurrence_detected: bool
    solver_success: bool
    solver_message: str

# ============================================================================
# DRUG SCHEDULE FUNCTIONS
# ============================================================================
def create_drug_schedule(regimen: str, dose_intensity: float = 1.0):
    """
    Create drug administration function based on clinical regimen
    
    Args:
        regimen: Treatment schedule type
        dose_intensity: Relative dose intensity (0.5-1.5)
    
    Returns:
        Function that takes time (days) and returns drug input rate (μM/day)
    """
    schedules = {
        "Carboplatin-Paclitaxel q21d": {
            "cycle_days": 21,
            "dose_duration": 1,  # 1-day infusion
            "dose_amount": 5.0 * dose_intensity
        },
        "Carboplatin-Paclitaxel q14d (dose-dense)": {
            "cycle_days": 14,
            "dose_duration": 1,
            "dose_amount": 4.0 * dose_intensity
        },
        "Pemetrexed q21d (non-squamous)": {
            "cycle_days": 21,
            "dose_duration": 0.5,
            "dose_amount": 3.5 * dose_intensity
        },
        "Weekly Paclitaxel (metronomic)": {
            "cycle_days": 7,
            "dose_duration": 0.5,
            "dose_amount": 2.0 * dose_intensity
        }
    }
    
    schedule = schedules.get(regimen, schedules["Carboplatin-Paclitaxel q21d"])
    
    def drug_input(t):
        """Pulsatile dosing function"""
        cycle_position = t % schedule["cycle_days"]
        if cycle_position < schedule["dose_duration"]:
            return schedule["dose_amount"]
        return 0.0
    
    return drug_input

# ============================================================================
# CACHED SIMULATION FUNCTION (PERFORMANCE OPTIMIZATION)
# ============================================================================
@st.cache_data(show_spinner=False, ttl=3600)
def run_simulation_cached(
    residual_burden: int,
    stage: str,
    histology: str,
    abc_score: float,
    plasticity_rate: float,
    epigenetic_noise: float,
    regimen: str,
    dose_intensity: float,
    simulation_days: int = 730,
    egfr_positive: bool = False,
    egfr_mutation_type: str = None
) -> SimulationResults:
    """
    Cached simulation runner - only recomputes when parameters change
    
    Returns SimulationResults dataclass
    """
    
    # Create patient profile
    patient = PatientProfile(
        stage=stage,
        histology=histology,
        residual_burden=residual_burden,
        baseline_plasticity=plasticity_rate,
        abc_expression=abc_score
    )
    
    # Get parameter dictionary
    patient_params = patient.to_params_dict()
    
    # Initialize models
    epigenetic_model = EpigeneticStateMachine(
        instability_sigma=epigenetic_noise,
        heritability_h=0.8
    )
    
    abc_model = ABCMediatedEfflux(
        basal_abcc1=abc_score,
        basal_abcg2=abc_score * 0.5
    )
    
    # Check if EGFR-mutant model should be used
    if egfr_positive and egfr_mutation_type:
        # Use EGFR mutation model
        egfr_model = EGFRMutationModel(mutation_type=egfr_mutation_type)
        
        # Create osimertinib schedule (80mg daily standard dose)
        osi_schedule = egfr_model.create_osimertinib_schedule(dose_mg=80)
        
        # Add osimertinib schedule to params
        patient_params['osi_dose_schedule'] = osi_schedule
        patient_params['egfr_mutation_rate'] = 1e-7  # Per cell per division
        patient_params['dtp_entry_rate'] = 0.001     # Persister entry rate
        patient_params['dtp_exit_rate'] = 0.0005     # Persister exit rate
        
        # Initial state for EGFR model: [S_EGFR, R_T790M, R_C797S, DTP, D_osi]
        initial_state = egfr_model.get_initial_state(residual_burden)
        
        # Define ODE wrapper for EGFR model
        def ode_wrapper(t, y):
            return egfr_model.osimertinib_ode(y, t, patient_params)
        
        # Event detection for EGFR model (sum all cell populations)
        def recurrence_event(t, y):
            return (y[0] + y[1] + y[2] + y[3]) - 1e8
    else:
        # Standard chemotherapy model
        # Create drug schedule
        drug_schedule_func = create_drug_schedule(regimen, dose_intensity)
        
        # Initial state: [S, R, D_eff, ABC_expr, E_score]
        initial_state = [
            residual_burden,  # Sensitive cells
            max(1.0, residual_burden * 0.002),  # Resistant cells (0.2% baseline per ITH data)
            0.0,  # Drug concentration
            abc_score,  # ABC expression
            epigenetic_noise  # Epigenetic instability baseline
        ]
        
        # Define ODE wrapper
        def ode_wrapper(t, y):
            """Wrapper for solve_ivp (note: t and y are swapped vs odeint)"""
            return nsclc_digital_twin_ode(
                y, t, patient_params, {}, 
                epigenetic_model, abc_model, drug_schedule_func
            )
        
        # Event detection: stop when recurrence threshold reached
        def recurrence_event(t, y):
            """Event function: triggers when total tumour burden exceeds 1e8"""
            return (y[0] + y[1]) - 1e8
    
    recurrence_event.terminal = True
    recurrence_event.direction = 1  # Only trigger when crossing upward
    
    # Solve ODE with event detection
    try:
        solution = solve_ivp(
            ode_wrapper,
            t_span=[0, simulation_days],
            y0=initial_state,
            method='LSODA',  # Stiff solver, best for biological systems
            dense_output=True,
            events=recurrence_event,
            rtol=1e-6,
            atol=1e-9,
            max_step=1.0  # Maximum 1 day step for dosing resolution
        )
        
        # Check if recurrence was detected
        recurrence_detected = len(solution.t_events[0]) > 0
        if recurrence_detected:
            recurrence_time_days = solution.t_events[0][0]
            recurrence_time_months = recurrence_time_days / 30.44
        else:
            recurrence_time_months = simulation_days / 30.44
        
        # Generate dense output for plotting
        t_eval = np.linspace(0, solution.t[-1], 1000)
        y_eval = solution.sol(t_eval)
        
        # Handle different model outputs
        if egfr_positive and egfr_mutation_type:
            # EGFR model: [S_EGFR, R_T790M, R_C797S, DTP, D_osi]
            # Map to standard output format for visualization
            results = SimulationResults(
                time=t_eval,
                sensitive_cells=np.maximum(0, y_eval[0, :]),  # S_EGFR
                resistant_cells=np.maximum(0, y_eval[1, :] + y_eval[2, :] + y_eval[3, :]),  # All resistant populations
                drug_concentration=np.maximum(0, y_eval[4, :]),  # Osimertinib (nM)
                abc_expression=np.ones_like(t_eval) * abc_score,  # Fixed ABC (not dynamic in EGFR model)
                epigenetic_score=np.ones_like(t_eval) * epigenetic_noise,  # Fixed
                recurrence_time=recurrence_time_months,
                recurrence_detected=recurrence_detected,
                solver_success=solution.success,
                solver_message=solution.message
            )
        else:
            # Standard chemotherapy model: [S, R, D_eff, ABC_expr, E_score]
            results = SimulationResults(
                time=t_eval,
                sensitive_cells=np.maximum(0, y_eval[0, :]),
                resistant_cells=np.maximum(0, y_eval[1, :]),
                drug_concentration=np.maximum(0, y_eval[2, :]),
                abc_expression=np.maximum(0, y_eval[3, :]),
                epigenetic_score=np.maximum(0, y_eval[4, :]),
                recurrence_time=recurrence_time_months,
                recurrence_detected=recurrence_detected,
                solver_success=solution.success,
                solver_message=solution.message
            )
        
    except Exception as e:
        # Return failed result
        t_dummy = np.linspace(0, simulation_days, 100)
        results = SimulationResults(
            time=t_dummy,
            sensitive_cells=np.zeros_like(t_dummy),
            resistant_cells=np.zeros_like(t_dummy),
            drug_concentration=np.zeros_like(t_dummy),
            abc_expression=np.zeros_like(t_dummy),
            epigenetic_score=np.zeros_like(t_dummy),
            recurrence_time=0.0,
            recurrence_detected=False,
            solver_success=False,
            solver_message=str(e)
        )
    
    return results

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def plot_tumour_dynamics(results: SimulationResults, treatment_type: str = "chemotherapy") -> go.Figure:
    """Main tumour population plot"""
    fig = go.Figure()
    
    time_months = results.time / 30.44
    total_cells = results.sensitive_cells + results.resistant_cells
    
    fig.add_trace(go.Scatter(
        x=time_months, 
        y=results.sensitive_cells,
        name='Sensitive Cells',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='%{y:.2e} cells<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=time_months,
        y=results.resistant_cells,
        name='Resistant Cells',
        line=dict(color='#d62728', width=2),
        hovertemplate='%{y:.2e} cells<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=time_months,
        y=total_cells,
        name='Total tumour Burden',
        line=dict(color='#2ca02c', width=3),
        hovertemplate='%{y:.2e} cells<extra></extra>'
    ))
    
    # Recurrence threshold
    fig.add_hline(
        y=1e8, 
        line_dash="dash", 
        line_color="rgba(255,0,0,0.5)",
        annotation_text="Clinical Recurrence Threshold (10⁸ cells)",
        annotation_position="right"
    )
    
    treatment_title = "Osimertinib (EGFR-TKI)" if treatment_type == "osimertinib" else "Maintenance Chemotherapy"
    fig.update_layout(
        title=f"tumour Population Dynamics Under {treatment_title}",
        xaxis_title="Time (months)",
        yaxis_title="Cell Count",
        yaxis_type="log",
        yaxis_range=[0, 11],  # 10^0 (1 cell) to 10^11
        hovermode='x unified',
        height=500,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
    )
    
    return fig

def plot_drug_and_abc(results: SimulationResults, treatment_type: str = "chemotherapy") -> go.Figure:
    """Drug concentration and ABC expression over time"""
    drug_title = "Osimertinib Concentration (Daily Dosing)" if treatment_type == "osimertinib" else "Drug Concentration (Pulsatile Dosing)"
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(drug_title, "ABC Transporter Expression"),
        vertical_spacing=0.15,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    time_months = results.time / 30.44
    
    # Drug concentration
    fig.add_trace(
        go.Scatter(
            x=time_months,
            y=results.drug_concentration,
            name='Drug Concentration',
            line=dict(color='#17becf', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(23,190,207,0.3)'
        ),
        row=1, col=1
    )
    
    # ABC expression
    fig.add_trace(
        go.Scatter(
            x=time_months,
            y=results.abc_expression,
            name='ABC Expression',
            line=dict(color='#ff7f0e', width=2)
        ),
        row=2, col=1
    )
    
    drug_unit = "nM" if treatment_type == "osimertinib" else "μM"
    fig.update_xaxes(title_text="Time (months)", row=2, col=1)
    fig.update_yaxes(title_text=f"Concentration ({drug_unit})", row=1, col=1)
    fig.update_yaxes(title_text="Relative Expression", row=2, col=1)
    
    fig.update_layout(height=600, showlegend=False, hovermode='x unified')
    
    return fig

def plot_epigenetic_trajectory(results: SimulationResults) -> go.Figure:
    """Epigenetic instability evolution"""
    fig = go.Figure()
    
    time_months = results.time / 30.44
    
    fig.add_trace(go.Scatter(
        x=time_months,
        y=results.epigenetic_score,
        name='Epigenetic Instability (σ²)',
        line=dict(color='#9467bd', width=3),
        fill='tozeroy',
        fillcolor='rgba(148,103,189,0.3)'
    ))
    
    fig.update_layout(
        title="Epigenetic Instability Accumulation Under Drug Pressure",
        xaxis_title="Time (months)",
        yaxis_title="Epigenetic Instability Score (σ²)",
        height=400,
        annotations=[
            dict(
                x=time_months[-1] * 0.6,
                y=results.epigenetic_score.max() * 0.7,
                text="Drug pressure → ↑ epigenetic noise → ↑ phenotypic switching",
                showarrow=True,
                arrowhead=2,
                ax=-80,
                ay=-40
            )
        ]
    )
    
    return fig

def plot_resistance_ratio(results: SimulationResults) -> go.Figure:
    """Resistant fraction over time"""
    fig = go.Figure()
    
    time_months = results.time / 30.44
    total_cells = results.sensitive_cells + results.resistant_cells
    resistant_fraction = results.resistant_cells / (total_cells + 1e-10)  # Avoid division by zero
    
    fig.add_trace(go.Scatter(
        x=time_months,
        y=resistant_fraction * 100,
        name='Resistant Fraction',
        line=dict(color='#e377c2', width=3),
        fill='tozeroy',
        fillcolor='rgba(227,119,194,0.3)'
    ))
    
    fig.update_layout(
        title="Evolution of Resistant Cell Fraction",
        xaxis_title="Time (months)",
        yaxis_title="Resistant Cells (%)",
        yaxis_range=[0, 100],
        height=400
    )
    
    return fig

def plot_resistance_fraction(results: SimulationResults) -> go.Figure:
    """Compatibility wrapper using the same implementation as plot_resistance_ratio."""
    return plot_resistance_ratio(results)

def display_recurrence_prediction(recurrence_time_months: float, detected: bool) -> None:
    """Show a high-level summary of recurrence prediction."""
    if detected:
        # Color coding based on recurrence time (months)
        if recurrence_time_months > 16:
            emoji = "🟢"
            label = "favorable"
        elif recurrence_time_months >= 10:
            emoji = "🟡"
            label = "intermediate"
        else:
            emoji = "🔴"
            label = "high-risk"

        st.markdown(
            f"""
            <div style="margin:1.5rem 0;">
              <h3 style="margin:0;">Predicted Clinical Recurrence {emoji}</h3>
              <p style="margin:0.25rem 0 0;">
                Estimated time to recurrence: <strong>{recurrence_time_months:.1f} months</strong>
                (<em>{label} risk</em>)<br/>
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info(f"🟢 No recurrence detected within simulation window (~{recurrence_time_months:.1f} months). This is clinically plausible and would explain the high rates of survival for EGFR+ patients given TKIs at {recurrence_time_months:.1f} months")

def display_parameter_importance() -> None:
    """Display qualitative notes about how key parameters affect recurrence."""
    st.subheader("📌 Qualitative Parameter Effects")
    st.markdown("""
    - **Higher ABC expression** → faster drug efflux → earlier risk of recurrence.
    - **Higher phenotypic plasticity (μ)** → more rapid S→R switching → earlier recurrence.
    - **Higher epigenetic instability (σ²)** → more non-genetic variability → broader resistance emergence.
    - **More dose-dense / metronomic regimens** can delay recurrence in highly plastic tumours.
    """)

# ============================================================================
# USER INTERFACE
# ============================================================================
def main():
    # Header
    st.title("🔬 NSCLC tumour Resistance & Recurrence Simulator")
    st.markdown("""
    **Mechanistic modeling of epigenetic plasticity, ABC transporter-mediated chemoresistance, and EGFR-TKI resistance**
    
    Based on:
    - Dhawan et al. *Nat Sci Rep* 2016 (doi:10.1038/srep28597) - Phenotypic switching
    - Lei et al. arXiv:1901.09747 2019 - Epigenetic plasticity
    - Fletcher et al. *Cancer Res* 2010 - ABC transporter kinetics
    - Ramalingam et al. *NEJM* 2020 - FLAURA trial (osimertinib efficacy)
    - Sharma et al. *Cell* 2010 - Drug-tolerant persisters
    """)
    
    # Sidebar: Patient Configuration
    st.sidebar.header("🏥 Patient Configuration")
    
    st.sidebar.subheader("Clinical Parameters")
    
    with st.sidebar.expander("ℹ️ About Clinical Parameters"):
        st.markdown("""
        **Pathologic Stage:** Anatomic extent per AJCC TNM classification.(1) Stage affects tumour carrying capacity:
        - IIA (T2bN0M0): K=10⁹ cells, 5-yr survival 60%
        - IIB (T2bN1/T3N0): K=5×10⁹ cells, 5-yr survival 53%
        - IIIA (T1-3N2/T4N0-1): K=10¹⁰ cells, 5-yr survival 36%
        - IIIB (T4N2): K=10¹¹ cells, 5-yr survival 26%
        
        **Histology:** Adenocarcinoma (doubling time 180-220d, growth rate 0.032-0.039 day⁻¹) vs. squamous (240-280d, 0.025-0.029 day⁻¹).(6,7) Adenocarcinoma responds better to pemetrexed (ORR 25% vs. 9%) due to lower thymidylate synthase expression.(9,11)
        
        **Residual Burden:** ctDNA-detected micrometastases post-R0 resection. Burden >10³ cells: HR 3.8 for recurrence.(16,17) MRD-positive patients: median DFS 8-12 months vs. >36 months in MRD-negative.(19,20)
        """, unsafe_allow_html=True)
    
    stage = st.sidebar.selectbox(
        "Pathologic Stage",
        ["IIA", "IIB", "IIIA", "IIIB"],
        index=2,
        help="AJCC TNM stage: IIA (60% 5-yr survival) to IIIB (26% 5-yr survival)"
    )
    
    histology = st.sidebar.selectbox(
        "Histology",
        ["adenocarcinoma", "squamous"],
        help="Adenocarcinoma: faster growth (0.035 d⁻¹), pemetrexed-sensitive. Squamous: slower growth (0.027 d⁻¹), pemetrexed-resistant due to 3-4× higher thymidylate synthase.(6,7,11)"
    )
    
    residual_burden = st.sidebar.slider(
        "Residual tumour Burden (cells)",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
        help="Micrometastases post-resection quantified by ctDNA. >10³ cells = HR 3.8 for recurrence (95% CI 2.1-6.9). Detection threshold: ~10⁸-10⁹ cells (0.5-1cm on CT).(16-18)"
    )
    
    st.sidebar.subheader("Molecular Markers")
    
    with st.sidebar.expander("ℹ️ About EGFR Mutations"):
        st.markdown("""
        **EGFR Mutations:** Activating alterations in tyrosine kinase domain conferring TKI sensitivity.(21) Prevalence: 15% Western, 35-50% Asian populations.(12,24)
        
        **Exon19del (E746-A750):** 45-50% of EGFR+ cases. Best prognosis: median PFS 19.1 months on osimertinib, OS 38.6 months. IC50 6-15 nM.(25,30,31,32)
        
        **L858R:** 40-45% of cases. Slightly inferior: median PFS 17.1 months, OS 31.2 months (HR 0.79 vs. exon19del). IC50 18-25 nM. Higher CNS progression rate (15% vs. 9%).(25,31,32,33)
        
        **T790M (Resistance):** Gatekeeper mutation in 50-60% of acquired TKI resistance. Increases ATP affinity, sterically blocks reversible TKIs. Osimertinib IC50 12 nM (200-fold selectivity over WT).(31,35,36)
        
        **FLAURA Trial:** Osimertinib 80mg daily: ORR 80%, median PFS 18.9 months, median OS 38.6 months.(25) Resistance mechanisms: C797S (10-15%), MET amplification (5-10%), drug-tolerant persisters.(26-28)
        """, unsafe_allow_html=True)
    
    # EGFR mutation status
    egfr_positive = st.sidebar.checkbox(
        "EGFR Mutation Positive",
        value=False,
        help="EGFR-activating mutations (15% Western, 35-50% Asian). Osimertinib: ORR 80%, median PFS 18.9 months (FLAURA trial).(25)"
    )
    
    egfr_mutation_type = None
    if egfr_positive:
        egfr_mutation_type = st.sidebar.selectbox(
            "EGFR Mutation Type",
            ["exon19del", "L858R", "T790M"],
            help="Exon19del (45%, best): PFS 19.1mo, OS 38.6mo. L858R (40%): PFS 17.1mo, OS 31.2mo. T790M: Resistance mutation, IC50 12nM.(25,30,31,32)"
        )
        st.sidebar.info("💊 **Osimertinib 80mg daily:** t½=48h, Css 300-500 nM, 95% protein-bound. Third-generation irreversible EGFR TKI.(25,31)")
    
    with st.sidebar.expander("ℹ️ About ABC Transporters"):
        st.markdown("""
        **ABC Transporters:** ATP-dependent efflux pumps (ABCB1/P-gp, ABCG2/BCRP) that extrude chemotherapy from cells.(38,39)
        
        **IHC Scoring:**
        - **0:** No staining, k_efflux = 0.01 d⁻¹
        - **1+:** Weak staining <10% cells, k_efflux = 0.05 d⁻¹  
        - **2+:** Moderate staining 10-50% cells, k_efflux = 0.10 d⁻¹
        - **3+:** Strong staining >50% cells, k_efflux = 0.20 d⁻¹
        
        **Clinical Impact (Meta-analysis, n=4,145):**(42)
        - High expression (IHC 2-3): ORR reduction OR 0.39 (95% CI 0.28-0.55)
        - PFS: HR 1.89 (95% CI 1.52-2.35) vs. low expression
        
        **Functional Effects:**(43,44)
        - ABCB1: Reduces carboplatin by 60-75% (EC50: 1.5→4.5-6.0 μM)
        - ABCG2: Reduces paclitaxel by 40-50%
        """, unsafe_allow_html=True)
    
    abc_score = st.sidebar.slider(
        "ABC Transporter Expression",
        min_value=0.0,
        max_value=3.0,
        value=1.0,
        step=0.1,
        format="%.1f",
        help="ABCB1/ABCG2 IHC score. High expression (2-3): 61% ORR reduction, PFS HR 1.89. Meta-analysis n=4,145 patients.(42) Reduces carboplatin IC50 from 1.5→6.0 μM.(43)"
    )
    
    st.sidebar.subheader("Epigenetic Parameters")
    
    with st.sidebar.expander("ℹ️ About Epigenetic Plasticity"):
        st.markdown("""
        **Phenotypic Plasticity (μ):** Non-genetic S→R transition rate via chromatin remodeling, independent of mutation.(46-48)
        
        **Single-cell lineage tracing:**(49,50)
        - Basal conditions: μ = 0.001-0.007 d⁻¹ (1-5% DTP entry in 7-14 days)
        - Under drug pressure: μ = 0.02-0.05 d⁻¹ (10-30% DTP)
        
        **Clinical Impact:**(52)
        - High chromatin remodeling: median PFS 8.2 months
        - Low remodeling: median PFS 15.7 months (HR 2.3, p<0.001)
        
        **Model Implementation:** dR/dt includes +μ·σ²·S term. Higher μ accelerates resistant flux.(53,65)
        
        ---
        
        **Epigenetic Instability (σ²):** Variance in heritable chromatin states during cell division.(55)
        
        **Inheritance Fidelity:**(56,57)
        - Genetic mutations: ~10⁻⁹ per base per division
        - Epigenetic states: ~10⁻³ to 10⁻² per locus (1000× higher error rate)
        - DNMT1 maintenance fidelity: 90-95%
        
        **Single-cell RNA-seq:**(59,60)
        - Chromatin accessibility CV: 0.15-0.85 across patients
        - High variance (σ² >1.0): 3.2× increased early relapse risk
        - Median survival: 9.1 vs. 18.7 months (p=0.003)
        
        **Drug-induced changes:**(63,64)
        - Chemotherapy increases transcriptional heterogeneity 2-3 fold
        - Model: dσ²/dt = k_drug·D·(σ_max - σ²)
        
        **Interaction:** Plasticity and instability multiply: μ·σ²·S. Requires both transition capacity (μ) and substrate heterogeneity (σ²).(65)
        """, unsafe_allow_html=True)
    
    plasticity_rate = st.sidebar.slider(
        "Phenotypic Plasticity Rate (μ)",
        min_value=0.01,
        max_value=0.5,
        value=0.12,
        step=0.01,
        format="%.2f",
        help="S→R transition rate. Range: 0.001-0.05 d⁻¹ (single-cell data). High plasticity: median PFS 8.2 vs. 15.7 months (HR 2.3).(49,50,52)"
    )
    
    epigenetic_noise = st.sidebar.slider(
        "Baseline Epigenetic Instability (σ²)",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1,
        format="%.1f",
        help="Chromatin state variance. Epigenetic error rate 10⁻³-10⁻² (1000× genetic mutation rate). High σ²: 3.2× relapse risk, survival 9.1 vs. 18.7mo.(56,57,59,60)"
    )
    
    st.sidebar.subheader("Treatment Protocol")
    
    with st.sidebar.expander("ℹ️ About Treatment Regimens"):
        st.markdown("""
        **Carboplatin-Paclitaxel q21d (Standard):**
        - Carboplatin AUC 5-6 + Paclitaxel 175-200 mg/m² IV every 21 days(66)
        - Adjuvant benefit: 5-yr survival +5.4% (HR 0.89, 95% CI 0.82-0.96)(67)
        - PK: Carboplatin peak 15-25 μM, t½ 2.5-6h, >95% cleared in 72h(68)
        - Paclitaxel: triphasic t½ (0.27h/2.1h/13.2h), prolonged tissue retention(69)
        - Neutrophil nadir: day 10-14; Platelet nadir: day 14-21(70)
        
        **q14d Dose-Dense:**
        - 14-day intervals minimize regrowth between cycles (dose-density hypothesis)(71)
        - Breast cancer: DFS HR 0.74 (p=0.01); NSCLC subset: favorable trends(72,73)
        - AUC increase: +35-40% over 12 weeks vs. q21d(74)
        - Toxicity: Grade 3-4 neutropenia 45% vs. 28% (p<0.001), requires G-CSF(75)
        - Model: shortens drug-free interval 18→11 days, sustains pressure on ABC 1-2 clones(76)
        
        **Pemetrexed q21d (Non-Squamous Only):**
        - 500 mg/m² IV q21d. Folate antimetabolite: inhibits TS, DHFR, GARFT(77,78)
        - Histology-selective: 3-4× higher TS in squamous = resistance(11,79)
        - Adenocarcinoma: median PFS 5.3 vs. 2.8 months squamous (p<0.001)(11)
        - Maintenance PFS benefit: +2.8 months (HR 0.62, p<0.001)(81)
        - PK: Css 150-250 μM, t½ 3.5h, 81% renal elimination(80)
        - Model EC50: 2.5 μM sensitive, 8-12 μM ABC-expressing(82)
        
        **Weekly Paclitaxel (Metronomic):**
        - 80-100 mg/m² weekly. Low-dose, high-frequency approach(83)
        - Grade 3-4 neutropenia: <10% vs. 30-45% with q21d(84)
        - Time-averaged concentration: +40% vs. q21d(85)
        - Anti-angiogenic effects at sublethal concentrations(86)
        - Non-inferior efficacy: ORR 24% vs. 28% (p=0.31), better QoL(87,88)
        - Model: drug-free interval 18→4 days, limits resistant expansion(89)
        """, unsafe_allow_html=True)
    
    regimen = st.sidebar.selectbox(
        "Maintenance Regimen",
        [
            "Carboplatin-Paclitaxel q21d",
            "Carboplatin-Paclitaxel q14d (dose-dense)",
            "Pemetrexed q21d (non-squamous)",
            "Weekly Paclitaxel (metronomic)"
        ],
        help="q21d: standard, +5.4% 5-yr survival.(67) q14d: DFS HR 0.74, +35-40% AUC.(72,74) Pemetrexed: +2.8mo PFS, non-squamous only.(81) Weekly: +40% time-avg concentration.(85)"
    )
    
    with st.sidebar.expander("ℹ️ About Dose Intensity"):
        st.markdown("""
        **Relative Dose Intensity (RDI):** Actual delivered dose as % of protocol-specified standard, accounting for reductions, delays, omissions.(90)
        
        **Meta-analysis Evidence:**(91,92)
        - RDI <85%: Inferior survival (HR 1.23 per 10% reduction, 95% CI 1.15-1.32)
        - NSCLC: RDI ≥80% = median OS 13.8 months vs. 9.2 months <80% (p<0.001)(93)
        
        **Dose Escalation (>100%):**
        - Minimal additional benefit at 125-150% RDI
        - Grade 3-4 AE rate doubles(94,95)
        
        **Pharmacodynamics:**(96)
        - Carboplatin: Sigmoidal dose-response, Hill coefficient 2-3
        - EC50 ~1.5-2.5 μM. Reductions below threshold disproportionately compromise efficacy
        - 70% RDI: peak 20→14 μM, approaches EC50_resistant (4-6 μM)(97,98)
        
        **Model Implementation:** D_max = RDI × D_standard directly scales achievable drug concentration.
        """, unsafe_allow_html=True)
    
    dose_intensity = st.sidebar.slider(
        "Relative Dose Intensity (%)",
        min_value=50,
        max_value=150,
        value=100,
        step=10,
        help="RDI <85%: HR 1.23 per 10% reduction (meta-analysis).(91,92) NSCLC: ≥80% = OS 13.8 vs. 9.2mo.(93) 70% RDI drops carboplatin from 20→14 μM, near EC50_resistant.(96-98)"
    ) / 100.0
    
    with st.sidebar.expander("ℹ️ About Simulation Duration"):
        st.markdown("""
        **Temporal Projection Window:** Duration of ODE integration to capture resistance evolution dynamics.(99)
        
        **Clinical Context:**
        - Median DFS stage II-III: 18-30 months(100,101)
        - 24-36 months captures 70-80% of recurrence events
        - Late recurrences (15-20%): dormant cells reactivating after quiescence(102,103)
        
        **Computational Considerations:**
        - 24 months: ~15,000 ODE evaluations
        - 48 months: ~30,000 evaluations
        - Error tolerances: rtol ≤10⁻⁶, atol ≤10⁻⁹(105)
        
        **Expected Recurrence Times:**(106)
        - High-risk (ABC ≥2.0, μ ≥0.2): 8-24 months
        - Low-risk (ABC <1.0, μ <0.1): >48 months
        
        **Recommendations:**
        - 6-12 months: Early resistance, chemo response assessment
        - 24-36 months: Standard DFS analysis
        - 36-48 months: Late recurrence patterns, persister reactivation(104)
        """, unsafe_allow_html=True)
    
    simulation_days = st.sidebar.slider(
        "Simulation Duration (months)",
        min_value=6,
        max_value=48,
        value=24,
        step=6,
        help="Temporal horizon for resistance evolution. Median DFS 18-30mo (stage II-III). 24-36mo captures 70-80% events. High-risk: 8-24mo. Low-risk: >48mo.(99-106)"
    ) * 30
    
    # Advanced options
    with st.sidebar.expander("⚙️ Advanced Options"):
        show_debug = st.checkbox("Show solver diagnostics", value=False)
        comparison_mode = st.checkbox("Enable comparison mode", value=False)
    
    # Run simulation button
    st.sidebar.markdown("---")
    run_button = st.sidebar.button("▶️ RUN SIMULATION", type="primary", use_container_width=True)
    
    # Main content area
    if run_button or 'results' in st.session_state:
        if run_button:
            spinner_text = "🔄 Running mechanistic ODE simulation..."
            if egfr_positive:
                spinner_text = "🔄 Running EGFR-TKI resistance simulation..."
            
            with st.spinner(spinner_text):
                results = run_simulation_cached(
                    residual_burden=residual_burden,
                    stage=stage,
                    histology=histology,
                    abc_score=abc_score,
                    plasticity_rate=plasticity_rate,
                    epigenetic_noise=epigenetic_noise,
                    regimen=regimen,
                    dose_intensity=dose_intensity,
                    simulation_days=simulation_days,
                    egfr_positive=egfr_positive,
                    egfr_mutation_type=egfr_mutation_type
                )
                st.session_state.results = results
        
        results = st.session_state.results
        
        # Check solver status
        if not results.solver_success:
            st.error(f"⚠️ Solver failed: {results.solver_message}")
            st.stop()
        
        # Display primary outcome
        display_recurrence_prediction(
            results.recurrence_time if results.recurrence_detected else simulation_days/30,
            detected=results.recurrence_detected
        )
        
        st.markdown("---")
        
        # Visualization tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 tumour Dynamics",
            "🧬 Epigenetic Evolution",
            "💊 Drug & ABC Kinetics",
            "📈 Resistance Fraction"
        ])
        
        with tab1:
            # Show initial conditions info
            if egfr_positive:
                initial_s = results.sensitive_cells[0]
                initial_r = results.resistant_cells[0]
                initial_total = initial_s + initial_r
                st.info(f"""
                **EGFR Model Initial Conditions (t=0):**
                - Sensitive (EGFR-mutant): {initial_s:.1f} cells ({initial_s/initial_total*100:.1f}%)
                - Resistant (T790M + C797S + DTP): {initial_r:.1f} cells ({initial_r/initial_total*100:.2f}%)
                - Total residual burden: {initial_total:.0f} cells
                """)
            
            treatment_type = "osimertinib" if egfr_positive else "chemotherapy"
            fig1 = plot_tumour_dynamics(results, treatment_type)
            st.plotly_chart(fig1, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                initial_sensitive = results.sensitive_cells[0]
                final_sensitive = results.sensitive_cells[-1]
                delta_s = final_sensitive - initial_sensitive
                st.metric("Final Sensitive Cells", f"{final_sensitive:.2e}", 
                         delta=f"{delta_s:.2e}", delta_color="inverse")
            with col2:
                initial_resistant = results.resistant_cells[0]
                final_resistant = results.resistant_cells[-1]
                delta_r = final_resistant - initial_resistant
                st.metric("Final Resistant Cells", f"{final_resistant:.2e}",
                         delta=f"{delta_r:.2e}")
            with col3:
                initial_total = initial_sensitive + initial_resistant
                total_burden = results.sensitive_cells[-1] + results.resistant_cells[-1]
                delta_total = total_burden - initial_total
                st.metric("Total tumour Burden", f"{total_burden:.2e}",
                         delta=f"{delta_total:.2e}")
        
        with tab2:
            fig2 = plot_epigenetic_trajectory(results)
            st.plotly_chart(fig2, use_container_width=True)
            
            st.info("""
            **Epigenetic Instability Dynamics:**
            - Drug pressure induces chromatin remodeling and epigenetic state changes
            - Higher σ² increases rate of phenotypic switching (S ↔ R)
            - Represents non-genetic, reversible resistance mechanism
            """)
        
        with tab3:
            fig3 = plot_drug_and_abc(results, treatment_type)
            st.plotly_chart(fig3, use_container_width=True)
            
            if egfr_positive:
                st.info(f"""
                **Osimertinib Pharmacokinetics ({egfr_mutation_type}):**
                - Daily oral dosing: 80mg once daily (standard FDA-approved dose)
                - Half-life: 48 hours (once-daily dosing achieves steady state)
                - Steady-state concentration: ~300-500 nM
                - EC50 for {egfr_mutation_type}: {15 if egfr_mutation_type=='exon19del' else 18 if egfr_mutation_type=='L858R' else 12} nM
                """)
            else:
                st.info("""
                **ABC Transporter Feedback Loop:**
                - Chemotherapy induces ABCB1/ABCG2 expression
                - Increased efflux reduces intracellular drug concentration
                - Creates adaptive resistance even without genetic mutations
                """)
        
        with tab4:
            fig4 = plot_resistance_fraction(results)
            st.plotly_chart(fig4, use_container_width=True)
        
        # Parameter importance section
        st.markdown("---")
        display_parameter_importance()
        
        # Debug info
        if show_debug:
            with st.expander("🔧 Solver Diagnostics"):
                st.write(f"**Solver Status:** {results.solver_message}")
                st.write(f"**Integration Points:** {len(results.time)}")
                st.write(f"**Time Range:** 0 to {results.time[-1]:.1f} days")
                st.write(f"**Final State:**")
                st.write(f"- Sensitive: {results.sensitive_cells[-1]:.2e}")
                st.write(f"- Resistant: {results.resistant_cells[-1]:.2e}")
                st.write(f"- Drug: {results.drug_concentration[-1]:.4f} μM")
                st.write(f"- ABC: {results.abc_expression[-1]:.2f}")
                st.write(f"- Epigenetic: {results.epigenetic_score[-1]:.2f}")
    
    else:
        st.info("👈 Configure patient parameters in the sidebar and click **RUN SIMULATION**")
        
        # Show example cases
        st.subheader("📋 Example Clinical Scenarios")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Favorable Prognosis:**
            - Stage IIA adenocarcinoma
            - Low ABC expression (0.5)
            - Low plasticity rate (0.05)
            - Standard q21d regimen
            
            *Expected: 30-48 month recurrence*
            """)
        
        with col2:
            st.markdown("""
            **Poor Prognosis:**
            - Stage IIIB squamous
            - High ABC expression (2.5)
            - High plasticity rate (0.3)
            - High epigenetic noise (1.5)
            
            *Expected: 8-14 month recurrence*
            """)

if __name__ == "__main__":
    main()

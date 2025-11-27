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
    simulation_days: int = 730
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
        """Event function: triggers when total tumor burden exceeds 1e8"""
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
def plot_tumor_dynamics(results: SimulationResults) -> go.Figure:
    """Main tumor population plot"""
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
        name='Total Tumor Burden',
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
    
    fig.update_layout(
        title="Tumor Population Dynamics Under Maintenance Chemotherapy",
        xaxis_title="Time (months)",
        yaxis_title="Cell Count",
        yaxis_type="log",
        yaxis_range=[1, 11],  # 10^1 to 10^11
        hovermode='x unified',
        height=500,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
    )
    
    return fig

def plot_drug_and_abc(results: SimulationResults) -> go.Figure:
    """Drug concentration and ABC expression over time"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Drug Concentration (Pulsatile Dosing)", "ABC Transporter Expression"),
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
    
    fig.update_xaxes(title_text="Time (months)", row=2, col=1)
    fig.update_yaxes(title_text="Concentration (μM)", row=1, col=1)
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
        st.info(f"🧮 No recurrence detected within simulation window (~{recurrence_time_months:.1f} months).")

def display_parameter_importance() -> None:
    """Display qualitative notes about how key parameters affect recurrence."""
    st.subheader("📌 Qualitative Parameter Effects")
    st.markdown("""
    - **Higher ABC expression** → faster drug efflux → earlier recurrence.
    - **Higher phenotypic plasticity (μ)** → more rapid S→R switching → earlier recurrence.
    - **Higher epigenetic instability (σ²)** → more non-genetic variability → broader resistance emergence.
    - **More dose-dense / metronomic regimens** can delay recurrence in highly plastic tumors.
    """)

# ============================================================================
# USER INTERFACE
# ============================================================================
def main():
    # Header
    st.title("🔬 NSCLC Tumor Resistance & Recurrence Simulator")
    st.markdown("""
    **Mechanistic modeling of epigenetic plasticity and ABC transporter-mediated chemoresistance**
    
    Based on:
    - Dhawan et al. *Nat Sci Rep* 2016 (doi:10.1038/srep28597) - Phenotypic switching
    - Lei et al. arXiv:1901.09747 2019 - Epigenetic plasticity
    - Fletcher et al. *Cancer Res* 2010 - ABC transporter kinetics
    """)
    
    # Sidebar: Patient Configuration
    st.sidebar.header("🏥 Patient Configuration")
    
    st.sidebar.subheader("Clinical Parameters")
    stage = st.sidebar.selectbox(
        "Pathologic Stage",
        ["IIA", "IIB", "IIIA", "IIIB"],
        index=2,
        help="Tumor stage after surgical resection"
    )
    
    histology = st.sidebar.selectbox(
        "Histology",
        ["adenocarcinoma", "squamous"],
        help="NSCLC subtype affects growth rate and drug sensitivity"
    )
    
    residual_burden = st.sidebar.slider(
        "Residual Tumor Burden (cells)",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
        help="Estimated microscopic residual disease after R0 resection (10²-10⁴ cells typical)"
    )
    
    st.sidebar.subheader("Molecular Markers")
    abc_score = st.sidebar.slider(
        "ABC Transporter Expression",
        min_value=0.0,
        max_value=3.0,
        value=1.0,
        step=0.1,
        format="%.1f",
        help="ABCB1/ABCG2 immunohistochemistry score (0=negative, 3=strong positive)"
    )
    
    st.sidebar.subheader("Epigenetic Parameters")
    plasticity_rate = st.sidebar.slider(
        "Phenotypic Plasticity Rate (μ)",
        min_value=0.01,
        max_value=0.5,
        value=0.12,
        step=0.01,
        format="%.2f",
        help="Rate of drug-induced S→R phenotypic switching (higher = faster adaptation)"
    )
    
    epigenetic_noise = st.sidebar.slider(
        "Baseline Epigenetic Instability (σ²)",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1,
        format="%.1f",
        help="Stochastic epigenetic state variability (higher = more heterogeneity)"
    )
    
    st.sidebar.subheader("Treatment Protocol")
    regimen = st.sidebar.selectbox(
        "Maintenance Regimen",
        [
            "Carboplatin-Paclitaxel q21d",
            "Carboplatin-Paclitaxel q14d (dose-dense)",
            "Pemetrexed q21d (non-squamous)",
            "Weekly Paclitaxel (metronomic)"
        ],
        help="Maintenance chemotherapy schedule"
    )
    
    dose_intensity = st.sidebar.slider(
        "Relative Dose Intensity (%)",
        min_value=50,
        max_value=150,
        value=100,
        step=10,
        help="Dose reduction/escalation (100% = standard dose)"
    ) / 100.0
    
    simulation_days = st.sidebar.slider(
        "Simulation Duration (months)",
        min_value=6,
        max_value=48,
        value=24,
        step=6,
        help="How long to project tumor dynamics"
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
            with st.spinner("🔄 Running mechanistic ODE simulation..."):
                results = run_simulation_cached(
                    residual_burden=residual_burden,
                    stage=stage,
                    histology=histology,
                    abc_score=abc_score,
                    plasticity_rate=plasticity_rate,
                    epigenetic_noise=epigenetic_noise,
                    regimen=regimen,
                    dose_intensity=dose_intensity,
                    simulation_days=simulation_days
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
            "📊 Tumor Dynamics",
            "🧬 Epigenetic Evolution",
            "💊 Drug & ABC Kinetics",
            "📈 Resistance Fraction"
        ])
        
        with tab1:
            fig1 = plot_tumor_dynamics(results)
            st.plotly_chart(fig1, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                final_sensitive = results.sensitive_cells[-1]
                st.metric("Final Sensitive Cells", f"{final_sensitive:.2e}")
            with col2:
                final_resistant = results.resistant_cells[-1]
                st.metric("Final Resistant Cells", f"{final_resistant:.2e}")
            with col3:
                total_burden = results.sensitive_cells[-1] + results.resistant_cells[-1]
                st.metric("Total Tumor Burden", f"{total_burden:.2e}")
        
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
            fig3 = plot_drug_and_abc(results)
            st.plotly_chart(fig3, use_container_width=True)
            
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

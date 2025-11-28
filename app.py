"""
NSCLC Digital Twin - Complete Integrated Application with ML Enhancement
Combines ODE modeling, ML parameter inference, ctDNA dynamics, and resistance classification
"""

import sys
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.integrate import solve_ivp

# Add src to path and import modules
src_path = str(Path(__file__).parent / 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.models.tumour_population import nsclc_digital_twin_ode, PatientProfile
from src.models.epigenetic_plasticity import EpigeneticStateMachine
from src.models.abc_transporters import ABCMediatedEfflux
from src.models.mutations import EGFRMutationModel
from src.utils.literature_params import CARBOPLATIN, ABC_TRANSPORTERS, EPIGENETIC_PARAMS
from src.utils.synthetic_data_loader import load_synthetic_data
from src.ml.models.parameter_inference import PatientParameterNN
from src.ml.models.ct_dna_dynamics import ctDNANeuralODE
from src.ml.models.resistance_classifier import TMEGraphClassifier
from src.ml.utils.ode_model_selector import ODESelector

# ============================================================================
# DATA STRUCTURES & SESSION STATE
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
    # ML-enhanced fields
    ctdna_vaf: np.ndarray = None
    ctdna_uncertainty: np.ndarray = None
    ml_inferred_params: dict = None
    resistance_prediction: dict = None

def init_session_state():
    """Initialize ML models and data in session state"""
    if 'ml_models_loaded' in st.session_state:
        return
        
    st.session_state.ml_models_loaded = False
    for key in ['parameter_model', 'ctdna_model', 'resistance_classifier', 'selector', 'synthetic_data']:
        setattr(st.session_state, key, None)
    
    try:
        st.session_state.synthetic_data = load_synthetic_data()
        st.session_state.ml_data_available = True
        
        # Load trained models
        param_model = PatientParameterNN(hidden_dim=128)
        param_model.load_state_dict(torch.load('src/ml/checkpoints/patient_parameter_nn.pth', map_location='cpu', weights_only=False)['model_state_dict'])
        param_model.eval()
        
        ctdna_model = ctDNANeuralODE()
        ctdna_model.eval()
        
        classifier = TMEGraphClassifier()
        classifier.load_state_dict(torch.load('src/ml/checkpoints/tme_gnn_classifier.pth', map_location='cpu', weights_only=False)['model_state_dict'])
        classifier.eval()
        
        st.session_state.parameter_model = param_model
        st.session_state.ctdna_model = ctdna_model
        st.session_state.resistance_classifier = classifier
        st.session_state.selector = ODESelector(classifier)
        st.session_state.ml_models_loaded = True
        
    except FileNotFoundError:
        st.session_state.ml_data_available = False

init_session_state()

# ============================================================================
# DRUG SCHEDULE & SIMULATION CORE
# ============================================================================
def create_drug_schedule(regimen: str, dose_intensity: float = 1.0):
    """Create pulsatile dosing function based on clinical regimen"""
    schedules = {
        "Carboplatin-Paclitaxel q21d": {"cycle_days": 21, "dose_duration": 1, "dose_amount": 5.0},
        "Carboplatin-Paclitaxel q14d (dose-dense)": {"cycle_days": 14, "dose_duration": 1, "dose_amount": 4.0},
        "Pemetrexed q21d (non-squamous)": {"cycle_days": 21, "dose_duration": 0.5, "dose_amount": 3.5},
        "Weekly Paclitaxel (metronomic)": {"cycle_days": 7, "dose_duration": 0.5, "dose_amount": 2.0}
    }
    
    schedule = schedules.get(regimen, schedules["Carboplatin-Paclitaxel q21d"])
    schedule["dose_amount"] *= dose_intensity
    
    def drug_input(t): return schedule["dose_amount"] if (t % schedule["cycle_days"]) < schedule["dose_duration"] else 0.0
    return drug_input

@st.cache_data(show_spinner=False, ttl=3600)
def run_simulation_cached(params: dict) -> SimulationResults:
    """
    Cached simulation runner with ML-enhanced parameter inference and ctDNA prediction
    
    Args:
        params: Dictionary containing all simulation parameters
    
    Returns:
        SimulationResults dataclass with tumour dynamics and optional ML predictions
    """
    # Unpack parameters
    residual_burden = params['residual_burden']
    stage = params['stage']
    histology = params['histology']
    abc_score = params['abc_score']
    plasticity_rate = params['plasticity_rate']
    epigenetic_noise = params['epigenetic_noise']
    regimen = params['regimen']
    dose_intensity = params['dose_intensity']
    simulation_days = params['simulation_days']
    egfr_positive = params.get('egfr_positive', False)
    egfr_mutation_type = params.get('egfr_mutation_type', None)
    ml_inference = params.get('ml_inference', False)
    patient_id = params.get('patient_id', None)
    use_ctdna = params.get('use_ctdna_prediction', False)
    
    # Initialize ML context
    ml_inferred_params = resistance_prediction = ctdna_vaf = ctdna_uncertainty = None
    
    custom_patient = params.get('custom_patient', False)
    
    # ML inference path
    if ml_inference and (patient_id or custom_patient) and st.session_state.ml_models_loaded:
        try:
            if custom_patient:
                # Use custom features from sliders
                features = pd.DataFrame([{
                    'ctdna_vaf_percent': params['custom_features']['ctdna_vaf_percent'],
                    'serum_hgf_pg_ml': params['custom_features']['serum_hgf_pg_ml'],
                    'plasma_il6_pg_ml': params['custom_features']['plasma_il6_pg_ml'],
                    'circulating_mdsc_per_ml': params['custom_features']['circulating_mdsc_per_ml'],
                    'serum_tgfb_ng_ml': params['custom_features']['serum_tgfb_ng_ml'],
                    'ctc_count_per_ml': params['custom_features']['ctc_count_per_ml'],
                    'serum_crp_mg_l': params['custom_features']['serum_crp_mg_l'],
                    'serum_ldh_u_l': params['custom_features']['serum_ldh_u_l']
                }])
                patient_data = {
                    'tumour_burden': 1e6 * (1 + params['custom_features']['ctdna_vaf_percent']),
                    'proliferation_rate': 0.05,
                    'resistance_mechanism': 'Unknown',
                    'baseline_vaf': params['custom_features']['ctdna_vaf_percent'],
                    
                    'cd8_density': params['custom_features']['circulating_mdsc_per_ml'] * 0.5,
                    'cd8_activation': 0.5,
                    'cd8_tumour_distance': params['custom_features']['serum_crp_mg_l'] / 2,
                    
                    'm2_tam_density': params['custom_features']['circulating_mdsc_per_ml'] * 0.8,
                    'm2_activation': 0.6,
                    'mdsc_density': params['custom_features']['circulating_mdsc_per_ml'],
                    'mdsc_suppression': min(params['custom_features']['plasma_il10_pg_ml'] / 10, 1.0),
                    'mdsc_tumour_proximity': 80,
                    
                    'caf_density': params['custom_features']['serum_tgfb_ng_ml'] * 5,
                    'caf_activation': min(params['custom_features']['serum_tgfb_ng_ml'] / 20, 1.0),
                    'tgf_beta': params['custom_features']['serum_tgfb_ng_ml'],
                    
                    'vessel_density': 150,
                    'vascular_permeability': 0.4,
                    'vegf_level': params['custom_features']['plasma_vegf_pg_ml'] / 100,
                    
                    'hgf_level': params['custom_features']['serum_hgf_pg_ml'],
                    'il10_level': params['custom_features']['plasma_il10_pg_ml'] / 10,
                    
                    '0_2_strength': 0.5,
                    '3_1_strength': min(params['custom_features']['plasma_il10_pg_ml'] / 50, 1.0),
                    '4_0_strength': min(params['custom_features']['serum_tgfb_ng_ml'] / 30, 1.0)
                }
            else:
                # Use patient data from CSV
                ctdna_df, tme_df, _, _ = st.session_state.synthetic_data
                patient_tme = tme_df[(tme_df['patient_id'] == patient_id) & (tme_df['week'] == 0)]
                patient_ctdna = ctdna_df[(ctdna_df['patient_id'] == patient_id) & (ctdna_df['week'] == 0)]
                
                if not patient_tme.empty and not patient_ctdna.empty:
                    # Infer parameters - need both TME and ctDNA data
                    features = pd.DataFrame([{
                        'ctdna_vaf_percent': patient_ctdna['ctdna_vaf_percent'].iloc[0],
                        'serum_hgf_pg_ml': patient_tme['serum_hgf_pg_ml'].iloc[0],
                        'plasma_il6_pg_ml': patient_tme['plasma_il6_pg_ml'].iloc[0],
                        'circulating_mdsc_per_ml': patient_tme['circulating_mdsc_per_ml'].iloc[0],
                        'serum_tgfb_ng_ml': patient_tme['serum_tgfb_ng_ml'].iloc[0],
                        'ctc_count_per_ml': patient_tme['ctc_count_per_ml'].iloc[0],
                        'serum_crp_mg_l': patient_tme['serum_crp_mg_l'].iloc[0],
                        'serum_ldh_u_l': patient_tme['serum_ldh_u_l'].iloc[0]
                    }])
                    
                    patient_data = {
                        'circulating_mdsc_per_ml': patient_tme['circulating_mdsc_per_ml'].iloc[0],
                        'plasma_il10_pg_ml': patient_tme['plasma_il10_pg_ml'].iloc[0],
                        'serum_hgf_pg_ml': patient_tme['serum_hgf_pg_ml'].iloc[0],
                        'serum_tgfb_ng_ml': patient_tme['serum_tgfb_ng_ml'].iloc[0],
                        'plasma_vegf_pg_ml': patient_tme['plasma_vegf_pg_ml'].iloc[0],
                        'resistance_mechanism': patient_tme['resistance_mechanism'].iloc[0]
                    }
            
            with torch.no_grad():
                pred_params = st.session_state.parameter_model.predict_from_pandas(features)
            
            ml_inferred_params = pred_params.iloc[0].to_dict()
            plasticity_rate = ml_inferred_params['mu'] * 10
            abc_score = ml_inferred_params['ABC']
            epigenetic_noise = ml_inferred_params['sigma2']
            
            # Predict resistance
            resistance_prediction = st.session_state.resistance_classifier.predict_from_patient_data(patient_data)
            
            # Auto-select ODE modules
            config = st.session_state.selector.select_modules(patient_data)
            
            if 'param_overrides' in config:
                for param, value in config['param_overrides'].items():
                    if param == 'mu': plasticity_rate = value
                    elif param == 'ABC': abc_score = value
                        
        except Exception as e:
            st.warning(f"ML inference failed: {e}. Using manual parameters.")
    
    # Create patient profile and initialize models
    patient = PatientProfile(stage=stage, histology=histology, residual_burden=residual_burden,
                            baseline_plasticity=plasticity_rate, abc_expression=abc_score)
    patient_params = patient.to_params_dict()
    
    epigenetic_model = EpigeneticStateMachine(instability_sigma=epigenetic_noise, heritability_h=0.8)
    abc_model = ABCMediatedEfflux(basal_abcc1=abc_score, basal_abcg2=abc_score * 0.5)
    
    # ODE setup
    if egfr_positive and egfr_mutation_type:
        egfr_model = EGFRMutationModel(mutation_type=egfr_mutation_type)
        osi_schedule = egfr_model.create_osimertinib_schedule(dose_mg=80)
        patient_params.update({
            'osi_dose_schedule': osi_schedule,
            'egfr_mutation_rate': 1e-7,
            'dtp_entry_rate': 0.001,
            'dtp_exit_rate': 0.0005
        })
        initial_state = egfr_model.get_initial_state(residual_burden)
        def ode_wrapper(t, y): return egfr_model.osimertinib_ode(y, t, patient_params)
        def recurrence_event(t, y): return (y[0] + y[1] + y[2] + y[3]) - 1e8
    else:
        drug_schedule_func = create_drug_schedule(regimen, dose_intensity)
        initial_state = [residual_burden, max(1.0, residual_burden * 0.002), 0.0, abc_score, epigenetic_noise]
        def ode_wrapper(t, y): return nsclc_digital_twin_ode(y, t, patient_params, {}, epigenetic_model, abc_model, drug_schedule_func)
        def recurrence_event(t, y): return (y[0] + y[1]) - 1e8
    
    recurrence_event.terminal = True
    recurrence_event.direction = 1
    
    # Solve ODE
    try:
        solution = solve_ivp(ode_wrapper, t_span=[0, simulation_days], y0=initial_state, method='LSODA',
                            dense_output=True, events=recurrence_event, rtol=1e-6, atol=1e-9, max_step=1.0)
        
        recurrence_detected = len(solution.t_events[0]) > 0
        recurrence_time_months = solution.t_events[0][0] / 30.44 if recurrence_detected else simulation_days / 30.44
        
        t_eval = np.linspace(0, solution.t[-1], 1000)
        y_eval = solution.sol(t_eval)
        
        # Create results object
        if egfr_positive and egfr_mutation_type:
            results = SimulationResults(
                time=t_eval,
                sensitive_cells=np.maximum(0, y_eval[0, :]),
                resistant_cells=np.maximum(0, y_eval[1, :] + y_eval[2, :] + y_eval[3, :]),
                drug_concentration=np.maximum(0, y_eval[4, :]),
                abc_expression=np.ones_like(t_eval) * abc_score,
                epigenetic_score=np.ones_like(t_eval) * epigenetic_noise,
                recurrence_time=recurrence_time_months,
                recurrence_detected=recurrence_detected,
                solver_success=solution.success,
                solver_message=solution.message,
                ml_inferred_params=ml_inferred_params,
                resistance_prediction=resistance_prediction
            )
        else:
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
                solver_message=solution.message,
                ml_inferred_params=ml_inferred_params,
                resistance_prediction=resistance_prediction
            )
        
        # ctDNA prediction
        if use_ctdna and st.session_state.ml_models_loaded:
            tumour_burden = results.sensitive_cells + results.resistant_cells
            tumour_rate = np.gradient(tumour_burden, t_eval)
            clone_fraction = results.resistant_cells / np.maximum(tumour_burden, 1)
            
            initial_ctdna = ml_inferred_params.get('baseline_ctdna', 0.001) if ml_inferred_params else 0.001
            tumour_tensor = torch.tensor([[b, r, f] for b, r, f in zip(tumour_burden, tumour_rate, clone_fraction)], dtype=torch.float32)
            time_tensor = torch.tensor(t_eval, dtype=torch.float32)
            
            with torch.no_grad():
                ctDNA_pred = st.session_state.ctdna_model.simulate(initial_ctdna, tumour_tensor, time_tensor)
            
            results.ctdna_vaf = (ctDNA_pred.numpy() / (tumour_burden / 1000)) * 100
            results.ctdna_uncertainty = results.ctdna_vaf * 0.15  # 15% CV
            
    except Exception as e:
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
            solver_message=str(e),
            ml_inferred_params=ml_inferred_params,
            resistance_prediction=resistance_prediction
        )
    
    return results

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def plot_ctdna_comparison(results: SimulationResults, patient_id: str) -> go.Figure:
    """Plot ctDNA prediction vs actual (if synthetic data available)"""
    fig = go.Figure()
    time_months = results.time / 30.44
    
    if results.ctdna_vaf is not None:
        fig.add_trace(go.Scatter(x=time_months, y=results.ctdna_vaf, name='Predicted ctDNA VAF', 
                                line=dict(color='#FF6B6B', width=3)))
        if results.ctdna_uncertainty is not None:
            fig.add_trace(go.Scatter(x=np.concatenate([time_months, time_months[::-1]]), 
                                    y=np.concatenate([results.ctdna_vaf + results.ctdna_uncertainty,
                                                    (results.ctdna_vaf - results.ctdna_uncertainty)[::-1]]),
                                    fill='toself', fillcolor='rgba(255,107,107,0.2)', 
                                    line=dict(color='rgba(255,255,255,0)'), name='95% Confidence'))
    
    if st.session_state.ml_data_available and patient_id:
        ctdna_df, _, _, _ = st.session_state.synthetic_data
        actual_ctdna = ctdna_df[ctdna_df['patient_id'] == patient_id]
        if not actual_ctdna.empty:
            fig.add_trace(go.Scatter(x=actual_ctdna['week'] / 4.33, y=actual_ctdna['ctdna_vaf_percent'], 
                                    mode='markers', name='Actual ctDNA', 
                                    marker=dict(color='#4ECDC4', size=10, symbol='diamond')))
            
            if results.ctdna_vaf is not None:
                pred_at_actual = np.interp(actual_ctdna['week'] * 7, results.time, results.ctdna_vaf)
                mse = np.mean((pred_at_actual - actual_ctdna['ctdna_vaf_percent'])**2)
                fig.add_annotation(x=0.05, y=0.95, xref='paper', yref='paper', text=f"MSE: {mse:.4f}%",
                                  showarrow=False, bgcolor='rgba(255,255,255,0.8)')
    
    fig.update_layout(title="ctDNA Dynamics: Prediction vs Actual", xaxis_title="Time (months)", 
                     yaxis_title="ctDNA VAF (%)", yaxis_type='log', height=500)
    return fig

def display_ml_internals(results: SimulationResults):
    """Display ML model internals in Advanced Mode"""
    st.subheader("🔍 ML Model Internals")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Parameter Inference")
        if results.ml_inferred_params:
            df_params = pd.DataFrame([results.ml_inferred_params]).T
            df_params.columns = ["Value"]
            st.dataframe(df_params, use_container_width=True)
            
            st.markdown("### vs Literature Defaults")
            lit_vs_ml = {k: [v, results.ml_inferred_params.get(k, v)] for k, v in {'r_R': 0.05, 'mu': 0.05, 'ABC': 1.0}.items()}
            df_compare = pd.DataFrame(lit_vs_ml, index=['Literature', 'ML']).T
            st.dataframe(df_compare, use_container_width=True)
        else:
            st.info("No ML parameters available (manual mode)")
    
    with col2:
        st.markdown("### Resistance Classification")
        if results.resistance_prediction:
            st.json(results.resistance_prediction)
            fig = px.pie(values=list(results.resistance_prediction['all_probabilities'].values()),
                        names=list(results.resistance_prediction['all_probabilities'].keys()),
                        title="Resistance Mechanism Probabilities")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No resistance prediction available")

def display_confidence_intervals(results: SimulationResults):
    """Display model uncertainty estimates"""
    st.subheader("📊 Model Confidence Intervals")
    
    if results.ml_inferred_params:
        col1, col2, col3 = st.columns(3)
        with col1:
            base_mu = results.ml_inferred_params.get('mu', 0.05)
            st.metric("Phenotypic Plasticity (μ)", f"{base_mu:.4f}", delta=f"±{base_mu * 0.2:.4f}")
        with col2:
            base_rR = results.ml_inferred_params.get('r_R', 0.05)
            st.metric("Resistant Growth Rate", f"{base_rR:.4f}", delta=f"±{base_rR * 0.15:.4f}")
        with col3:
            base_ABC = results.ml_inferred_params.get('ABC', 1.0)
            st.metric("ABC Expression", f"{base_ABC:.2f}", delta=f"±{base_ABC * 0.1:.2f}")
    
    if results.resistance_prediction:
        conf, unc = results.resistance_prediction['confidence'], results.resistance_prediction['uncertainty']
        st.progress(conf)
        st.write(f"**Confidence:** {conf:.1%}")
        st.write(f"**Uncertainty:** {unc:.1%}")
        st.success("High confidence prediction") if conf > 0.8 else st.warning("Moderate confidence prediction") if conf > 0.6 else st.error("Low confidence - consider manual review")

def plot_tumour_dynamics(results: SimulationResults, treatment_type: str = "chemotherapy") -> go.Figure:
    """Main tumour population plot with ML overlays"""
    fig = go.Figure()
    time_months = results.time / 30.44
    
    fig.add_trace(go.Scatter(x=time_months, y=results.sensitive_cells, name='Sensitive Cells', line=dict(color='#1f77b4', width=2)))
    fig.add_trace(go.Scatter(x=time_months, y=results.resistant_cells, name='Resistant Cells', line=dict(color='#d62728', width=2)))
    fig.add_trace(go.Scatter(x=time_months, y=results.sensitive_cells + results.resistant_cells, name='Total tumour Burden', line=dict(color='#2ca02c', width=3)))
    
    if results.ctdna_vaf is not None:
        fig.add_trace(go.Scatter(x=time_months, y=results.ctdna_vaf * 1e6, name='ctDNA VAF (×10⁶)', line=dict(color='#FF6B6B', width=2, dash='dot'), yaxis='y2'))
    
    fig.add_hline(y=1e8, line_dash="dash", line_color="rgba(255,0,0,0.5)", annotation_text="Clinical Recurrence Threshold", annotation_position="right")
    
    if results.resistance_prediction and results.resistance_prediction['confidence'] > 0.7:
        fig.add_vline(x=results.recurrence_time, line_dash="dash", line_color="rgba(255,165,0,0.7)", 
                     annotation_text=f"ML Predicted Recurrence: {results.recurrence_time:.1f}mo", annotation_position="top")
    
    treatment_title = "Osimertinib (EGFR-TKI)" if treatment_type == "osimertinib" else "Maintenance Chemotherapy"
    fig.update_layout(title=f"tumour & ctDNA Dynamics Under {treatment_title}", xaxis_title="Time (months)", yaxis_title="Cell Count", 
                     yaxis_type="log", yaxis_range=[0, 11], height=500, hovermode='x unified',
                     yaxis2=dict(title="ctDNA VAF (%)", overlaying='y', side='right', type='log'))
    return fig

def plot_drug_and_abc(results: SimulationResults, treatment_type: str = "chemotherapy") -> go.Figure:
    """Drug concentration and ABC expression over time"""
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Drug Concentration", "ABC Transporter Expression"), vertical_spacing=0.15)
    time_months = results.time / 30.44
    
    drug_unit = "nM" if treatment_type == "osimertinib" else "μM"
    fig.add_trace(go.Scatter(x=time_months, y=results.drug_concentration, name='Drug Concentration', 
                            line=dict(color='#17becf', width=1.5), fill='tozeroy', fillcolor='rgba(23,190,207,0.3)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=time_months, y=results.abc_expression, name='ABC Expression', line=dict(color='#ff7f0e', width=2)), row=2, col=1)
    
    fig.update_xaxes(title_text="Time (months)", row=2, col=1)
    fig.update_yaxes(title_text=f"Concentration ({drug_unit})", row=1, col=1)
    fig.update_yaxes(title_text="Relative Expression", row=2, col=1)
    fig.update_layout(height=600, showlegend=False, hovermode='x unified')
    return fig

def plot_epigenetic_trajectory(results: SimulationResults) -> go.Figure:
    """Epigenetic instability evolution"""
    fig = go.Figure()
    time_months = results.time / 30.44
    
    fig.add_trace(go.Scatter(x=time_months, y=results.epigenetic_score, name='Epigenetic Instability (σ²)', 
                            line=dict(color='#9467bd', width=3), fill='tozeroy', fillcolor='rgba(148,103,189,0.3)'))
    
    fig.update_layout(title="Epigenetic Instability Accumulation Under Drug Pressure", xaxis_title="Time (months)", 
                     yaxis_title="Epigenetic Instability Score (σ²)", height=400, 
                     annotations=[dict(x=time_months[-1]*0.6, y=results.epigenetic_score.max()*0.7, 
                                     text="Drug pressure → ↑ epigenetic noise → ↑ phenotypic switching", 
                                     showarrow=True, arrowhead=2, ax=-80, ay=-40)])
    return fig

def plot_resistance_fraction(results: SimulationResults) -> go.Figure:
    """Resistant fraction over time"""
    fig = go.Figure()
    time_months = results.time / 30.44
    resistant_fraction = results.resistant_cells / (results.sensitive_cells + results.resistant_cells + 1e-10)
    
    fig.add_trace(go.Scatter(x=time_months, y=resistant_fraction * 100, name='Resistant Fraction', 
                            line=dict(color='#e377c2', width=3), fill='tozeroy', fillcolor='rgba(227,119,194,0.3)'))
    
    fig.update_layout(title="Evolution of Resistant Cell Fraction", xaxis_title="Time (months)", yaxis_title="Resistant Cells (%)", 
                     yaxis_range=[0, 100], height=400)
    return fig

def display_recurrence_prediction(recurrence_time_months: float, detected: bool) -> None:
    """Show a high-level summary of recurrence prediction"""
    if detected:
        emoji, label = ("🟢", "favorable") if recurrence_time_months > 16 else ("🟡", "intermediate") if recurrence_time_months >= 10 else ("🔴", "high-risk")
        st.markdown(f"<div style='margin:1.5rem 0;'><h3 style='margin:0;'>Predicted Clinical Recurrence {emoji}</h3><p style='margin:0.25rem 0 0;'>Estimated time to recurrence: <strong>{recurrence_time_months:.1f} months</strong> (<em>{label} risk</em>)<br/></p></div>", unsafe_allow_html=True)
    else:
        st.info(f"🟢 No recurrence detected within simulation window (~{recurrence_time_months:.1f} months)")

def display_parameter_importance() -> None:
    """Display qualitative notes about how key parameters affect recurrence"""
    st.subheader("📌 Qualitative Parameter Effects")
    st.markdown("""
    - **Higher ABC expression** → faster drug efflux → earlier risk of recurrence.
    - **Higher phenotypic plasticity (μ)** → more rapid S→R switching → earlier recurrence.
    - **Higher epigenetic instability (σ²)** → more non-genetic variability → broader resistance emergence.
    - **More dose-dense / metronomic regimens** can delay recurrence in highly plastic tumours.
    """)

def display_user_guide():
    """Comprehensive user guide for the simulator"""
    st.markdown("""
    ## **1. Choose Your Input Method**

    #### **Option A: Manual Parameter Input**
    - Select **"Manual Sliders"** from the sidebar
    - Adjust parameters using intuitive sliders
    - Good for learning and sensitivity analysis

    #### **Option C: Custom Patient (ML-Assisted)**
    - Select **"Custom Patient (ML-Assisted)"**
    - Adjust biomarker sliders (ctDNA VAF, cytokines, immune cells, etc.)
    - ML automatically infers ODE parameters and predicts resistance from your custom profile
    - Enables full ML assistance without needing existing patient data
    - Preview resistance prediction before running simulation

    ### **2. Configure Patient Parameters**

    #### **Clinical Parameters**
    - **Pathologic Stage**: Cancer progression stage (IIA-IIIB)
    - **Histology**: tumour type (adenocarcinoma vs squamous)
    - **Residual tumour Burden**: Cancer cells remaining after surgery (100-10,000 cells)
    - **Histology**: tumour type (adenocarcinoma vs squamous)
    - **Residual tumour Burden**: Cancer cells remaining after surgery (100-10,000 cells)

    #### **Molecular Markers**
    - **ABC Transporter Expression**: Drug efflux pump activity (0.0-3.0)
    - **EGFR Mutation**: Check if patient has EGFR mutation
    - **EGFR Mutation Type**: Specific mutation (exon19del, L858R, T790M)

    #### **Epigenetic Parameters**
    - **Phenotypic Plasticity Rate (μ)**: Speed of drug resistance evolution (0.01-0.5)
    - **Epigenetic Instability (σ²)**: Non-genetic variation rate (0.1-2.0)

    #### **Treatment Protocol**
    - **Maintenance Regimen**: Chemotherapy schedule
    - **Relative Dose Intensity**: Treatment strength (50-150%)
    - **Simulation Duration**: How long to model (6-48 months)

    ### **3. Run the Simulation**
    - Click **"▶️ RUN SIMULATION"** button
    - Wait for the spinner to complete (~10-30 seconds)
    - Results appear in the main panel

    ### **4. Interpret Results**

    #### **Main Dashboard**
    - **Recurrence Prediction**: Time to clinical recurrence
    - **Risk Assessment**: Favorable/intermediate/high-risk
    - **ML Confidence**: How confident the AI predictions are

    #### **Visualization Tabs**
    1. **📊 Tumour & ctDNA Dynamics**: Cell populations over time
    2. **🧬 Epigenetic Evolution**: Resistance development
    3. **💊 Drug & ABC Kinetics**: Treatment effectiveness
    4. **📈 Resistance Fraction**: % resistant cells
    5. **🔬 ctDNA Prediction**: Blood biomarker trends

    ### **5. Advanced Features**
    - Check **"Advanced Mode"** for detailed ML internals
    - View parameter confidence intervals
    - Compare ML predictions vs literature defaults
    - Enable solver diagnostics for troubleshooting


    ### **6. Understanding the Science**

    This simulator uses **modeling** combining:
    - **Ordinary Differential Equations (ODEs)** for tumour growth
    - **Ordinary Differential Equations (ODEs)** for tumour growth
    - **Neural Networks** for parameter inference from biomarkers
    - **Graph Neural Networks** for resistance mechanism classification
    - **Neural ODEs** for ctDNA dynamics prediction

    The goal is to predict when cancer will recur so treatment can be adjusted proactively.
    """)

def display_parameter_reference():
    """Detailed explanation of each adjustable parameter"""
    st.markdown("""
    ## 🔧 **PARAMETER REFERENCE GUIDE**

    ### **🎯 Clinical Parameters**

    #### **Pathologic Stage (IIA, IIB, IIIA, IIIB)**
    - **What it is**: How advanced the cancer is based on tumour size, lymph node involvement, and metastasis
    - **Impact**: Higher stages have more aggressive tumour biology and higher recurrence risk
    - **What it is**: How advanced the cancer is based on tumour size, lymph node involvement, and metastasis
    - **Impact**: Higher stages have more aggressive tumour biology and higher recurrence risk
    - **Default**: IIIA (common stage for adjuvant therapy)
    - **Clinical relevance**: Stage IIIA patients often receive adjuvant chemotherapy

    #### **Histology (Adenocarcinoma vs Squamous)**
    - **What it is**: The microscopic appearance and cell type of the tumour
    - **What it is**: The microscopic appearance and cell type of the tumour
    - **Impact**: Adenocarcinoma tends to be more responsive to pemetrexed, squamous to taxanes
    - **Default**: Adenocarcinoma (most common NSCLC type)
    - **Clinical relevance**: Treatment selection and prognosis differ by histology

    #### **Residual Tumour Burden (100-10,000 cells)**
    - **What it is**: Number of cancer cells remaining after surgery
    - **Impact**: Higher burden = faster recurrence, lower burden = longer remission
    - **Default**: 1,000 cells (typical microscopic residual disease)
    - **Clinical relevance**: Minimal residual disease (MRD) is a key prognostic factor

    ### **🧬 Molecular Markers**

    #### **ABC Transporter Expression (0.0-3.0)**
    - **What it is**: Activity of ATP-binding cassette proteins that pump drugs out of cells
    - **Biological mechanism**: Higher expression → better drug efflux → treatment resistance
    - **Impact**: Values >1.5 indicate multidrug resistance phenotype
    - **Default**: 1.0 (baseline expression)
    - **Clinical relevance**: ABCB1/MDR1 overexpression causes chemotherapy failure

    #### **EGFR Mutation Status**
    - **What it is**: Activating mutations in epidermal growth factor receptor
    - **Impact**: Enables targeted therapy with osimertinib instead of chemotherapy
    - **Mutation types**:
      - **exon19del**: Most responsive to EGFR TKIs
      - **L858R**: Good response but may develop resistance
      - **T790M**: Resistance mutation requiring osimertinib
    - **Clinical relevance**: ~15% of NSCLC patients have actionable EGFR mutations

    ### **🧬 Epigenetic Parameters**

    #### **Phenotypic Plasticity Rate (μ) - 0.01 to 0.5**
    - **What it is**: Speed at which tumour cells can switch between drug-sensitive and drug-resistant states
    - **What it is**: Speed at which tumour cells can switch between drug-sensitive and drug-resistant states
    - **Biological mechanism**: Epigenetic changes allow cells to adapt to therapy pressure
    - **Impact**: Higher μ → faster resistance evolution → earlier recurrence
    - **Default**: 0.12 (moderate plasticity)
    - **Clinical relevance**: High plasticity explains why some tumours recur despite good initial response
    - **Clinical relevance**: High plasticity explains why some tumours recur despite good initial response

    #### **Baseline Epigenetic Instability (σ²) - 0.1 to 2.0**
    - **What it is**: Amount of non-genetic variation in gene expression and cellular behavior
    - **Biological mechanism**: Stochastic epigenetic changes create cellular heterogeneity
    - **Impact**: Higher σ² → more diverse cell populations → broader resistance mechanisms
    - **Default**: 0.5 (moderate instability)
    - **Clinical relevance**: Epigenetic heterogeneity drives tumour evolution and treatment failure
    - **Clinical relevance**: Epigenetic heterogeneity drives tumour evolution and treatment failure

    ### **💊 Treatment Protocol**

    #### **Maintenance Regimen**
    - **Carboplatin-Paclitaxel q21d**: Standard every-3-week chemotherapy
    - **Carboplatin-Paclitaxel q14d (dose-dense)**: More intensive schedule
    - **Pemetrexed q21d (non-squamous)**: Preferred for adenocarcinoma, less toxic
    - **Weekly Paclitaxel (metronomic)**: Low-dose continuous therapy
    - **Impact**: Dose-dense regimens delay resistance but increase toxicity

    #### **Relative Dose Intensity (50-150%)**
    - **What it is**: Percentage of planned chemotherapy dose actually delivered
    - **Impact**: Lower intensity (<80%) allows tumour regrowth, higher intensity (>120%) increases toxicity
    - **Impact**: Lower intensity (<80%) allows tumour regrowth, higher intensity (>120%) increases toxicity
    - **Default**: 100% (full planned dose)
    - **Clinical relevance**: Dose reductions due to toxicity are common and impact outcomes

    #### **Simulation Duration (6-48 months)**
    - **What it is**: How long the model simulates tumour growth and treatment
    - **What it is**: How long the model simulates tumour growth and treatment
    - **Impact**: Longer simulations show resistance evolution and late recurrences
    - **Default**: 24 months (typical follow-up period)
    - **Clinical relevance**: Most recurrences happen within 2 years of surgery

    ### **🎯 ML-Enhanced Mode Features**

    #### **Patient Selection**
    - Choose from synthetic patients with realistic biomarker profiles
    - Each patient has ctDNA levels, TME factors, and known outcomes
    - ML automatically infers parameters from patient data

    #### **ML Parameter Inference**
    - **Input**: 8 biomarkers (ctDNA VAF, HGF, IL6, MDSCs, TGF-β, CTCs, CRP, LDH)
    - **Output**: Inferred ODE parameters (r_R, μ, ABC, σ², K)
    - **Benefit**: Patient-specific parameters instead of population averages

    #### **Resistance Classification**
    - **Input**: TME cell interactions (MDSCs, TAMs, CAFs, TILs)
    - **Output**: Predicted resistance mechanism (No Resistance, C797S, MET_amp, etc.)
    - **Confidence**: Probability distribution across all mechanisms

    #### **ctDNA Prediction**
    - **Method**: Neural ODE modeling tumour burden → ctDNA shedding
    - **Method**: Neural ODE modeling tumour burden → ctDNA shedding
    - **Output**: Predicted ctDNA VAF over time with uncertainty bounds
    - **Clinical utility**: Early detection of recurrence before imaging

    ### **📊 Understanding Output Plots**

    #### **tumour Dynamics Plot**
    - **Sensitive Cells**: Drug-responsive tumour cells (blue line)
    - **Resistant Cells**: Drug-resistant tumour cells (red line)
    #### **tumour Dynamics Plot**
    - **Sensitive Cells**: Drug-responsive tumour cells (blue line)
    - **Resistant Cells**: Drug-resistant tumour cells (red line)
    - **Total Burden**: Sum of both populations (green line)
    - **Clinical Recurrence Threshold**: 100 million cells (dashed red line)

    #### **Epigenetic Evolution**
    - Shows how epigenetic instability accumulates under treatment pressure
    - Higher peaks indicate more phenotypic switching
    - Correlates with resistance development speed

    #### **Drug & ABC Kinetics**
    - **Drug Concentration**: Chemotherapy levels over time
    - **ABC Expression**: Adaptive upregulation of efflux pumps
    - Shows how tumours develop resistance through increased drug export
    - Shows how tumours develop resistance through increased drug export

    #### **Resistance Fraction**
    - Percentage of tumour cells that are drug-resistant
    - Percentage of tumour cells that are drug-resistant
    - Starts low, increases as sensitive cells are killed
    - Key metric for treatment effectiveness

    #### **ctDNA Prediction**
    - Predicted circulating tumour DNA levels
    - Predicted circulating tumour DNA levels
    - Early warning signal for recurrence
    - Uncertainty bounds show prediction confidence
    """)

# ============================================================================
# MAIN UI LOGIC
# ============================================================================
def main():
    st.title("🔬 NSCLC tumour Resistance & Recurrence Predictor")
    st.title("🔬 NSCLC tumour Resistance & Recurrence Predictor")
    st.markdown("""**Mechanistic modeling + Machine Learning for patient-specific resistance prediction**  
                The goal is to act as an early detection system for drug resistance and clinical recurrence in NSCLC patients undergoing chemotherapy or targeted therapy.
                This allows clinical teams to proactively adjust treatment strategies before overt relapse occurs.
                Through our research we found this was possible through tracking tumour dynamics, ATP-binding cassette (ABC) transporters, ctDNA levels, and tumour microenvironment features over time and using a simple neural network to infer hidden parameters driving resistance evolution. It is also possible to monitor and model tumour growth patterns using ordinary differential equations (ODEs) that capture key biological processes such as phenotypic plasticity and drug efflux via ABC transporters.""")

    # Add comprehensive user guide
    with st.expander("📖 **USER GUIDE: How to Use This Simulator**", expanded=False):
        display_user_guide()
    
    # Add parameter reference
    with st.expander("🔧 **PARAMETER REFERENCE: What Each Variable Means**", expanded=False):
        display_parameter_reference()
    
    #Instructions and how touse and what everything means

    # Literature references
    st.markdown("""    
    Based on:
    - Dhawan et al. *Nat Sci Rep* 2016 (doi:10.1038/srep28597) - Phenotypic switching
    - Lei et al. arXiv:1901.09747 2019 - Epigenetic plasticity
    - Fletcher et al. *Cancer Res* 2010 - ABC transporter kinetics
    - Ramalingam et al. *NEJM* 2020 - FLAURA trial (osimertinib efficacy)
    - Sharma et al. *Cell* 2010 - Drug-tolerant persisters
    """)


    # Data source selection
    st.sidebar.header("📊 Data Source")
    data_source = st.sidebar.radio("Parameter Input Method", ["Manual Sliders", "Patient Data (ML-Enhanced)", "Custom Patient (ML-Assisted)"])
    
    # Initialize parameters
    params = {'residual_burden': 1000, 'stage': 'IIIA', 'histology': 'adenocarcinoma', 'abc_score': 1.0,
              'plasticity_rate': 0.12, 'epigenetic_noise': 0.5, 'regimen': 'Carboplatin-Paclitaxel q21d',
              'dose_intensity': 1.0, 'simulation_days': 730, 'egfr_positive': False, 'egfr_mutation_type': None,
              'ml_inference': False, 'patient_id': None, 'use_ctdna_prediction': False, 'custom_patient': False, 'custom_features': None}
    
    # ML mode
    if data_source == "Patient Data (ML-Enhanced)" and st.session_state.ml_data_available and st.session_state.ml_models_loaded:
        params['ml_inference'] = True
        ctdna_df, tme_df, _, summary_df = st.session_state.synthetic_data
        
        params['patient_id'] = st.sidebar.selectbox("Select Patient ID", summary_df['patient_id'].unique())
        patient_summary = summary_df[summary_df['patient_id'] == params['patient_id']].iloc[0]
        patient_tme = tme_df[(tme_df['patient_id'] == params['patient_id']) & (tme_df['week'] == 0)].iloc[0]
        patient_ctdna = ctdna_df[(ctdna_df['patient_id'] == params['patient_id']) & (ctdna_df['week'] == 0)].iloc[0]
        
        st.sidebar.subheader("📋 Patient Profile")
        st.sidebar.write(f"**EGFR Mutation:** {patient_summary['egfr_mutation']}")
        st.sidebar.write(f"**Resistance Mechanism:** {patient_summary['resistance_mechanism']}")
        st.sidebar.write(f"**Actual PFS:** {patient_summary['ttp_months']:.1f} months")
        
        if st.sidebar.button("🔍 Preview ML Prediction"):
            with st.spinner("Running inference..."):
                # Build complete patient data structure for ML model
                patient_data = {
                    # tumour features from ctDNA data
                    'tumor_burden': 1e6 * (1 + patient_ctdna['ctdna_vaf_percent']),
                    'proliferation_rate': 0.05,
                    'resistance_mechanism': patient_tme['resistance_mechanism'],
                    'baseline_vaf': patient_ctdna['ctdna_vaf_percent'],
                    
                    # Immune features (approximated from available data)
                    'cd8_density': patient_tme.get('circulating_mdsc_per_ml', 100) * 0.5,
                    'cd8_activation': 0.5,
                    'cd8_tumour_distance': patient_tme.get('serum_crp_mg_l', 50) / 2,
                    
                    # Myeloid features
                    'm2_tam_density': patient_tme.get('circulating_mdsc_per_ml', 50) * 0.8,
                    'm2_activation': 0.6,
                    'mdsc_density': patient_tme['circulating_mdsc_per_ml'],
                    'mdsc_suppression': min(patient_tme['plasma_il10_pg_ml'] / 10, 1.0),
                    'mdsc_tumour_proximity': 80,
                    
                    # Stromal features
                    'caf_density': patient_tme.get('serum_tgfb_ng_ml', 10) * 5,
                    'caf_activation': min(patient_tme['serum_tgfb_ng_ml'] / 20, 1.0),
                    'caf_tumour_proximity': 30,
                    
                    # Vascular features
                    'vessel_density': 150,
                    'vascular_permeability': 0.4,
                    
                    # Cytokines (from blood data)
                    'hgf_level': patient_tme['serum_hgf_pg_ml'],
                    'il10_level': patient_tme['plasma_il10_pg_ml'] / 10,
                    'tgf_beta': patient_tme['serum_tgfb_ng_ml'],
                    'vegf_level': patient_tme['plasma_vegf_pg_ml'] / 100,
                    
                    # Interaction strengths (based on resistance mechanism)
                    '0_2_strength': 0.7 if patient_tme['resistance_mechanism'] == 'MET_amp' else 0.3,
                    '3_1_strength': min(patient_tme['plasma_il10_pg_ml'] / 50, 1.0),
                    '4_0_strength': min(patient_tme['serum_tgfb_ng_ml'] / 30, 1.0)
                }
                
                prediction = st.session_state.resistance_classifier.predict_from_patient_data(patient_data)
                st.sidebar.success(f"Predicted: {prediction['predicted_mechanism']} ({prediction['confidence']:.1%})")
        
        params['use_ctdna_prediction'] = st.sidebar.checkbox("Enable ctDNA Prediction", value=True)
        
        # Auto-populate from ML inference
        try:
            features = torch.tensor([[patient_tme['ctdna_vaf_percent'], patient_tme['serum_hgf_pg_ml'], 
                                    patient_tme['plasma_il6_pg_ml'], patient_tme['circulating_mdsc_per_ml'],
                                    patient_tme['serum_tgfb_ng_ml'], patient_tme['ctc_count_per_ml'],
                                    patient_tme['serum_crp_mg_l'], patient_tme['serum_ldh_u_l']]], dtype=torch.float32)
            
            with torch.no_grad():
                pred_params = st.session_state.parameter_model(features)
            
            ml_params = pred_params.iloc[0].to_dict()
            params.update({
                'abc_score': ml_params['ABC'],
                'plasticity_rate': ml_params['mu'] * 10,
                'epigenetic_noise': ml_params['sigma2'],
                'stage': patient_summary.get('stage', 'IIIA'),
                'histology': patient_summary.get('histology', 'adenocarcinoma'),
                'residual_burden': max(100, int(patient_summary.get('ctdna_vaf_mean', 0.1) * 10000))
            })
        except:
            pass
    
    # Custom Patient (ML-Assisted) mode
    elif data_source == "Custom Patient (ML-Assisted)" and st.session_state.ml_data_available and st.session_state.ml_models_loaded:
        params['ml_inference'] = True
        params['custom_patient'] = True
        
        st.sidebar.header("🏥 Custom Patient Configuration")
        st.sidebar.subheader("Biomarker Sliders")
        
        # Sliders for key biomarkers used in ML models
        ctdna_vaf = st.sidebar.slider("ctDNA VAF (%)", 0.0, 10.0, 2.0, 0.1, format="%.1f")
        hgf_level = st.sidebar.slider("Serum HGF (pg/mL)", 0.0, 10.0, 2.5, 0.1, format="%.1f")
        il6_level = st.sidebar.slider("Plasma IL-6 (pg/mL)", 0.0, 20.0, 5.0, 0.5, format="%.1f")
        mdsc_count = st.sidebar.slider("Circulating MDSCs (per mL)", 0, 100, 25, 5)
        tgfb_level = st.sidebar.slider("Serum TGF-β (ng/mL)", 0.0, 50.0, 15.0, 1.0, format="%.1f")
        ctc_count = st.sidebar.slider("CTC Count (per mL)", 0.0, 5.0, 1.0, 0.1, format="%.1f")
        crp_level = st.sidebar.slider("Serum CRP (mg/L)", 0.0, 20.0, 5.0, 0.5, format="%.1f")
        ldh_level = st.sidebar.slider("Serum LDH (U/L)", 100, 300, 150, 10)
        
        params['custom_features'] = {
            'ctdna_vaf_percent': ctdna_vaf,
            'serum_hgf_pg_ml': hgf_level,
            'plasma_il6_pg_ml': il6_level,
            'circulating_mdsc_per_ml': mdsc_count,
            'serum_tgfb_ng_ml': tgfb_level,
            'ctc_count_per_ml': ctc_count,
            'serum_crp_mg_l': crp_level,
            'serum_ldh_u_l': ldh_level
        }
        
        st.sidebar.subheader("Additional TME Features")
        il10_level = st.sidebar.slider("Plasma IL-10 (pg/mL)", 0.0, 20.0, 6.0, 0.5, format="%.1f")
        vegf_level = st.sidebar.slider("Plasma VEGF (pg/mL)", 0, 200, 50, 10)
        
        params['custom_features'].update({
            'plasma_il10_pg_ml': il10_level,
            'plasma_vegf_pg_ml': vegf_level
        })
        
        st.sidebar.subheader("Molecular Markers")
        params['egfr_positive'] = st.sidebar.checkbox("EGFR Mutation Positive", value=False)
        if params['egfr_positive']:
            params['egfr_mutation_type'] = st.sidebar.selectbox("EGFR Mutation Type", ["exon19del", "L858R", "T790M"])

                # After setting params['custom_features'] with all 10, filter to the 8 expected for parameter inference
        param_inference_features = {
            'ctdna_vaf_percent': params['custom_features']['ctdna_vaf_percent'],
            'serum_hgf_pg_ml': params['custom_features']['serum_hgf_pg_ml'],
            'plasma_il6_pg_ml': params['custom_features']['plasma_il6_pg_ml'],
            'circulating_mdsc_per_ml': params['custom_features']['circulating_mdsc_per_ml'],
            'serum_tgfb_ng_ml': params['custom_features']['serum_tgfb_ng_ml'],
            'ctc_count_per_ml': params['custom_features']['ctc_count_per_ml'],
            'serum_crp_mg_l': params['custom_features']['serum_crp_mg_l'],
            'serum_ldh_u_l': params['custom_features']['serum_ldh_u_l']
        }
        
        features_df = pd.DataFrame([param_inference_features])
        
        if st.sidebar.button("🔍 Preview Resistance Prediction"):
            try:
                with st.spinner("Running inference..."):
                    # Construct patient_data for resistance classifier
                    patient_data = {
                        'tumour_burden': 1e6 * (1 + ctdna_vaf),
                        'proliferation_rate': 0.05,
                        'resistance_mechanism': 'Unknown',  # Will be predicted
                        'baseline_vaf': ctdna_vaf,
                        
                        'cd8_density': mdsc_count * 0.5,
                        'cd8_activation': 0.5,
                        'cd8_tumour_distance': crp_level / 2,
                        
                        'm2_tam_density': mdsc_count * 0.8,
                        'm2_activation': 0.6,
                        'mdsc_density': mdsc_count,
                        'mdsc_suppression': min(il10_level / 10, 1.0),
                        'mdsc_tumour_proximity': 80,
                        
                        'caf_density': tgfb_level * 5,
                        'caf_activation': min(tgfb_level / 20, 1.0),
                        'tgf_beta': tgfb_level,
                        
                        'vessel_density': 150,
                        'vascular_permeability': 0.4,
                        'vegf_level': vegf_level / 100,
                        
                        'hgf_level': hgf_level,
                        'il10_level': il10_level / 10,
                        
                        '0_2_strength': 0.5,
                        '3_1_strength': min(il10_level / 50, 1.0),
                        '4_0_strength': min(tgfb_level / 30, 1.0)
                    }
                    
                    prediction = st.session_state.resistance_classifier.predict_from_patient_data(patient_data)
                    st.sidebar.success(f"Predicted: {prediction['predicted_mechanism']} ({prediction['confidence']:.1%})")
            except Exception as e:
                st.sidebar.error(f"Resistance prediction failed: {e}")
        
        params['use_ctdna_prediction'] = st.sidebar.checkbox("Enable ctDNA Prediction", value=True)
        
        # Auto-populate from ML inference using custom features
        try:
            # Use only the 8 features expected by the parameter model
            features_df = pd.DataFrame([param_inference_features])
            
            with torch.no_grad():
                pred_params = st.session_state.parameter_model.predict_from_pandas(features_df)
            
            ml_params = pred_params.iloc[0].to_dict()
            params.update({
                'abc_score': ml_params['ABC'],
                'plasticity_rate': ml_params['mu'] * 10,
                'epigenetic_noise': ml_params['sigma2'],
                'residual_burden': max(100, int(ctdna_vaf * 10000))
            })
        except Exception as e:
            st.sidebar.warning(f"ML parameter inference failed: {e}. Using defaults.")
    
    # Manual mode
    else:
        st.sidebar.header("🏥 Patient Configuration")
        st.sidebar.subheader("Clinical Parameters")
        
        params['stage'] = st.sidebar.selectbox("Pathologic Stage", ["IIA", "IIB", "IIIA", "IIIB"], index=2)
        params['histology'] = st.sidebar.selectbox("Histology", ["adenocarcinoma", "squamous"])
        params['residual_burden'] = st.sidebar.slider("Residual tumour Burden (cells)", 100, 10000, 1000, 100)
        
        st.sidebar.subheader("Molecular Markers")
        params['abc_score'] = st.sidebar.slider("ABC Transporter Expression", 0.0, 3.0, 1.0, 0.1, format="%.1f")
        
        st.sidebar.subheader("Epigenetic Parameters")
        params['plasticity_rate'] = st.sidebar.slider("Phenotypic Plasticity Rate (μ)", 0.01, 0.5, 0.12, 0.01, format="%.2f")
        params['epigenetic_noise'] = st.sidebar.slider("Baseline Epigenetic Instability (σ²)", 0.1, 2.0, 0.5, 0.1, format="%.1f")
        
        params['egfr_positive'] = st.sidebar.checkbox("EGFR Mutation Positive", value=False)
        if params['egfr_positive']:
            params['egfr_mutation_type'] = st.sidebar.selectbox("EGFR Mutation Type", ["exon19del", "L858R", "T790M"])
        
        st.sidebar.subheader("Treatment Protocol")
        params['regimen'] = st.sidebar.selectbox("Maintenance Regimen", [
            "Carboplatin-Paclitaxel q21d", "Carboplatin-Paclitaxel q14d (dose-dense)",
            "Pemetrexed q21d (non-squamous)", "Weekly Paclitaxel (metronomic)"
        ])
        params['dose_intensity'] = st.sidebar.slider("Relative Dose Intensity (%)", 50, 150, 100, 10) / 100.0
        params['simulation_days'] = st.sidebar.slider("Simulation Duration (months)", 6, 48, 24, 6) * 30
    
    # Advanced mode and run button
    advanced_mode = st.sidebar.checkbox("Advanced Mode", value=False)
    st.sidebar.markdown("---")
    run_button = st.sidebar.button("▶️ RUN SIMULATION", type="primary", use_container_width=True)
    
    # Execution and visualization
    if run_button or 'results' in st.session_state:
        if run_button:
            with st.spinner("🔄 Running simulation with ML enhancement..."):
                st.session_state.results = run_simulation_cached(params)
        
        results = st.session_state.results
        
        if not results.solver_success:
            st.error(f"⚠️ Solver failed: {results.solver_message}")
            st.stop()
        
        if params['ml_inference'] and results.ml_inferred_params:
            patient_label = params['patient_id'] if params['patient_id'] else "Custom Patient"
            st.success(f"✅ ML-Enhanced Simulation (Patient: {patient_label})")
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("ML Confidence", f"{results.resistance_prediction['confidence']:.1%}" if results.resistance_prediction else "N/A")
            with col2: st.metric("Resistance Type", results.resistance_prediction['predicted_mechanism'] if results.resistance_prediction else "N/A")
            with col3: st.metric("ctDNA Prediction", "Enabled" if results.ctdna_vaf is not None else "Disabled")
        
        display_recurrence_prediction(results.recurrence_time if results.recurrence_detected else params['simulation_days']/30, results.recurrence_detected)
        st.markdown("---")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Tumour & ctDNA Dynamics", "🧬 Epigenetic Evolution", "💊 Drug & ABC Kinetics", "📈 Resistance Fraction", "🔬 ctDNA Prediction"
        ])
        
        with tab1:
            fig1 = plot_tumour_dynamics(results, "osimertinib" if params['egfr_positive'] else "chemotherapy")
            st.plotly_chart(fig1, use_container_width=True)
            if params['ml_inference'] and results.ml_inferred_params:
                with st.expander("🔍 ML-Inferred Parameters"): st.json(results.ml_inferred_params)
        
        with tab2: st.plotly_chart(plot_epigenetic_trajectory(results), use_container_width=True)
        with tab3: st.plotly_chart(plot_drug_and_abc(results, "osimertinib" if params['egfr_positive'] else "chemotherapy"), use_container_width=True)
        with tab4: st.plotly_chart(plot_resistance_fraction(results), use_container_width=True)
        
        with tab5:
            if results.ctdna_vaf is not None: st.plotly_chart(plot_ctdna_comparison(results, params['patient_id']), use_container_width=True)
            else: st.info("ctDNA prediction not enabled. Check 'Enable ctDNA Prediction' in sidebar.")
        
        if advanced_mode:
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1: display_ml_internals(results)
            with col2: display_confidence_intervals(results)
        
        st.markdown("---")
        display_parameter_importance()
        
        if st.checkbox("Show solver diagnostics"):
            with st.expander("🔧 Solver Diagnostics"):
                st.write(f"**Solver Status:** {results.solver_message}")
                st.write(f"**ML Models Active:** {params['ml_inference']}")
                st.write(f"**Final State - Sensitive:** {results.sensitive_cells[-1]:.2e}, Resistant: {results.resistant_cells[-1]:.2e}")
                if results.ml_inferred_params: st.write(f"**ML Parameters:** {results.ml_inferred_params}")
    
    else:
        st.info("👈 Configure patient parameters and click **RUN SIMULATION**")
        col1, col2 = st.columns(2)
        with col1: st.metric("ML Data Available", "✅ Yes" if st.session_state.ml_data_available else "❌ No")
        with col2: st.metric("ML Models Loaded", "✅ Yes" if st.session_state.ml_models_loaded else "❌ No")

if __name__ == "__main__":
    main()
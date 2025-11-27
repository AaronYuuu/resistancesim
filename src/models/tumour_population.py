

def nsclc_digital_twin_ode(state, t, patient_params, treatment_params, 
                           epigenetic_model, abc_model, drug_schedule_func):
    """
    Main ODE function that gets passed to scipy.integrate.odeint/solve_ivp
    
    5-State System:
    [S, R, D_eff, ABC_expr, E_score]
    
    Based on:
    - Dhawan et al. Nat Sci Rep 2016 (doi:10.1038/srep28597) - phenotypic switching
    - Lei et al. arXiv:1901.09747 2019 - epigenetic plasticity
    - Fletcher et al. Cancer Res 2010 - ABC transporter kinetics
    
    Inputs:
    - state: [S, R, D_eff, ABC_expr, E_score]
    - t: current time (days)
    - patient_params: dictionary of patient-specific constants
    - treatment_params: dosing schedule parameters
    - epigenetic_model: instance of EpigeneticStateMachine
    - abc_model: instance of ABCMediatedEfflux
    - drug_schedule_func: function(t) -> drug input rate
    
    Returns:
    - [dSdt, dRdt, dDdt, dABCdt, dEscore_dt]
    """
    import numpy as np
    
    # Unpack state variables with bounds checking
    S = max(0, state[0])  # Sensitive cells
    R = max(0, state[1])  # Resistant cells
    D_eff = max(0, min(50.0, state[2]))  # Effective drug concentration (μM)
    ABC_expr = max(0, min(10.0, state[3]))  # ABC transporter expression
    E_score = max(0, min(3.0, state[4]))  # Epigenetic instability score
    
    # Extract parameters
    r_S = patient_params['growth_rate_sensitive']  # day^-1
    r_R = patient_params['growth_rate_resistant']  # day^-1
    K = patient_params['carrying_capacity']  # cells
    d = patient_params['death_rate']  # baseline death rate
    
    # Epigenetic parameters
    sigma = E_score  # Current epigenetic instability
    mu = patient_params['plasticity_rate']  # User plasticity index (0.05-0.3)
    mu_rate = mu * 0.001  # Convert to actual switching rate (/day): 5e-5 to 3e-4
    
    # Drug parameters
    EC50_S = patient_params['ec50_sensitive']  # μM
    EC50_R = patient_params['ec50_resistant']  # μM
    hill_coeff = patient_params.get('hill_coefficient', 2.0)
    
    # ABC transporter parameters
    Vmax = patient_params['vmax_abc'] * ABC_expr  # Scale by expression level
    Km_abc = patient_params['km_abc']
    
    # --- ABC-mediated efflux (Michaelis-Menten) ---
    # Fletcher et al. Cancer Res 2010 - P-gp mediated efflux
    efflux_rate = (Vmax * D_eff) / (Km_abc + D_eff)
    
    # --- Drug kill terms (Hill equation) ---
    # Dhawan et al. 2016 - dose-response relationship
    kill_S = (D_eff**hill_coeff) / (EC50_S**hill_coeff + D_eff**hill_coeff)
    kill_R = (D_eff**hill_coeff) / (EC50_R**hill_coeff + D_eff**hill_coeff)
    
    # --- Phenotypic switching (drug-induced) ---
    # Lei et al. 2019 - epigenetic state transitions
    # Switching is SLOW - happens over weeks, not days
    # High plasticity = faster switching but also metabolic cost
    switch_S_to_R = mu_rate * sigma * S * (1 + 0.5 * D_eff / (EC50_S + D_eff))  # Drug-induced switching
    switch_R_to_S = 0.5 * mu_rate * sigma * R  # Spontaneous reversion (half rate)
    
    # --- Logistic growth with competition ---
    # Standard tumor growth model with carrying capacity
    N_total = S + R
    growth_factor = 1 - (N_total / K)
    
    # --- dS/dt: Sensitive cell population ---
    # High epigenetic instability increases drug sensitivity (Lei 2019)
    sigma_sensitivity = 1.0 / (1.0 + sigma * 0.1)  # Higher sigma → slightly more sensitive
    dSdt = (
        r_S * S * growth_factor  # Logistic growth
        - d * S  # Baseline death
        - kill_S * patient_params['max_kill_rate'] * S / sigma_sensitivity  # Drug kill
        - switch_S_to_R  # Transition to resistant
        + switch_R_to_S  # Reversion from resistant
    )
    
    # --- dR/dt: Resistant cell population ---
    # ABC expression reduces drug effectiveness dramatically
    # High plasticity helps resistant cells adapt and survive
    # High sigma destabilizes ABC protection (epigenetic instability reduces transporter function)
    abc_base_protection = 1.0 / (1.0 + (ABC_expr * 3)**2)  # Strong quadratic: ABC=2.5 → 2% effectiveness (50x resistance)
    abc_growth_bonus = 1.0 + (ABC_expr**1.5 * 0.20)  # Superlinear: ABC=2.5 → 1.63x growth rate
    
    # Net ABC protection (sigma/plasticity affect switching dynamics, not drug protection)
    effective_protection = abc_base_protection
    
    dRdt = (
        r_R * R * growth_factor * abc_growth_bonus  # Growth with ABC bonus
        - d * R  # Baseline death
        - kill_R * patient_params['max_kill_rate'] * R * effective_protection  # Effective ABC protection
        + switch_S_to_R  # Gain from sensitive
        - switch_R_to_S  # Loss to sensitive
    )
    
    # --- dD/dt: Effective drug concentration ---
    # Input from dosing schedule, clearance, and ABC efflux
    drug_input = drug_schedule_func(t)  # Pulsatile dosing
    drug_clearance = patient_params['drug_clearance_rate'] * D_eff  # First-order
    
    dDdt = (
        drug_input  # Dosing
        - drug_clearance  # Systemic clearance
        - efflux_rate * 0.01  # ABC efflux (scaled for systemic concentration)
    )
    
    # --- dABC/dt: ABC transporter expression ---
    # Drug exposure induces ABC expression (adaptive resistance)
    # Saturable induction with decay
    abc_induction_rate = patient_params.get('abc_induction_rate', 0.1)
    abc_decay_rate = patient_params.get('abc_decay_rate', 0.05)
    abc_max = patient_params.get('abc_max_expression', 5.0)
    
    induction = abc_induction_rate * D_eff * (abc_max - ABC_expr) / (2.0 + D_eff)
    decay = abc_decay_rate * ABC_expr
    
    dABCdt = induction - decay
    
    # --- dE/dt: Epigenetic instability score ---
    # Accumulates under drug pressure, saturates, slow decay during recovery
    # Lei et al. 2019 - drug-induced epigenetic remodeling
    sigma_baseline = patient_params['baseline_sigma']
    sigma_max = patient_params.get('sigma_max', 2.5)
    epigenetic_induction = 0.05 * D_eff * (sigma_max - E_score) / (1.0 + D_eff)
    epigenetic_decay = 0.02 * (E_score - sigma_baseline)  # Relaxation to baseline
    
    dEscore_dt = epigenetic_induction - epigenetic_decay
    
    return [dSdt, dRdt, dDdt, dABCdt, dEscore_dt]

class PatientProfile:
    """Encapsulates all patient-specific biological parameters"""
    
    def __init__(self, stage, histology, residual_burden, 
                 baseline_plasticity, abc_expression):
        self.stage = stage
        self.histology = histology
        self.residual_burden = residual_burden
        self.baseline_plasticity = baseline_plasticity
        self.abc_expression = abc_expression
        
    def get_growth_rate(self):
        """Returns histology and ABC-level specific growth rate from literature_params.py"""
        from utils.literature_params import GROWTH_RATES
        
        # Select growth rate based on ABC expression level (correlates with aggressiveness)
        if self.histology in GROWTH_RATES:
            if self.abc_expression < 0.8:
                return GROWTH_RATES[self.histology].get('low', 0.020)
            elif self.abc_expression < 1.5:
                return GROWTH_RATES[self.histology].get('medium', 0.022)
            else:
                return GROWTH_RATES[self.histology].get('high', 0.032)
        return 0.022  # Default growth rate
    
    def get_carrying_capacity(self):
        """Returns stage-specific carrying capacity"""
        from utils.literature_params import CARRYING_CAPACITY
        
        stage_key = f"stage_{self.stage.replace('I', 'I')[:3]}"  # Normalize stage
        return CARRYING_CAPACITY.get(stage_key, 5e9)
    
    def to_params_dict(self):
        """Convert patient profile to parameter dictionary for ODE solver"""
        from utils.literature_params import CARBOPLATIN, ABC_TRANSPORTERS, EPIGENETIC_PARAMS
        
        return {
            'growth_rate_sensitive': self.get_growth_rate(),
            'growth_rate_resistant': self.get_growth_rate(),  # Same base rate
            'carrying_capacity': self.get_carrying_capacity(),
            'death_rate': 0.008,  # baseline apoptosis ~1% per day (Dhawan 2016)
            'plasticity_rate': self.baseline_plasticity,
            'baseline_sigma': EPIGENETIC_PARAMS['default_sigma'],
            'sigma_max': 2.5,
            'ec50_sensitive': CARBOPLATIN['ec50_sensitive'],
            'ec50_resistant': CARBOPLATIN['ec50_resistant'],
            'hill_coefficient': 2.0,
            'max_kill_rate': 0.8,  # Max kill rate at saturation (Dhawan 2016: 70-90% over 48hr)
            'drug_clearance_rate': CARBOPLATIN['clearance_rate'] * 24,  # Convert hr^-1 to day^-1 (was incorrectly divided)
            'vmax_abc': ABC_TRANSPORTERS['Vmax_abcc1'],
            'km_abc': ABC_TRANSPORTERS['Km_carboplatin'],
            'abc_induction_rate': ABC_TRANSPORTERS['induction_per_cycle'],
            'abc_decay_rate': 0.05,
            'abc_max_expression': 5.0,
        }
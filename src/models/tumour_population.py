

def nsclc_digital_twin_ode(state, t, patient_params, treatment_params, 
                           epigenetic_model, abc_model, drug_schedule_func):
    """
    Enhanced ODE function with realistic PK/PD modeling
    
    7-State System:
    [S, R, D_plasma, D_tumor, D_intra, ABC_expr, E_score]
    
    Includes:
    - Plasma drug compartment
    - Tumor extracellular compartment  
    - Intracellular drug compartment (ABC-affected)
    - Proper drug distribution and efflux kinetics
    """
    import numpy as np
    from .chemotherapy_pkpd import CarboplatinPKPD, PaclitaxelPKPD
    
    # Unpack state variables with bounds checking
    S = max(0, state[0])  # Sensitive cells
    R = max(0, state[1])  # Resistant cells
    D_plasma = max(0, min(100.0, state[2]))  # Plasma drug concentration (μM)
    D_tumor = max(0, min(50.0, state[3]))   # Tumor extracellular drug (μM)
    D_intra = max(0, min(25.0, state[4]))   # Intracellular drug (μM)
    ABC_expr = max(0, min(10.0, state[5]))  # ABC transporter expression
    E_score = max(0, min(3.0, state[6]))    # Epigenetic instability score
    
    # Extract parameters
    r_S = patient_params['growth_rate_sensitive']  # day^-1
    r_R = patient_params['growth_rate_resistant']  # day^-1
    K = patient_params['carrying_capacity']  # cells
    d = patient_params['death_rate']  # baseline death rate
    
    # Epigenetic parameters
    sigma = E_score  # Current epigenetic instability
    mu = patient_params['plasticity_rate']  # User plasticity index (0.05-0.3)
    mu_rate = mu * 0.001  # Convert to actual switching rate (/day): 5e-5 to 3e-4
    
    # Drug parameters - use intracellular concentration for kill
    EC50_S = patient_params['ec50_sensitive']  # μM
    EC50_R = patient_params['ec50_resistant']  # μM
    hill_coeff = patient_params.get('hill_coefficient', 2.0)
    
    # PK/PD parameters
    drug_type = patient_params.get('drug_type', 'carboplatin')
    if drug_type == 'carboplatin':
        drug_model = CarboplatinPKPD()
        k_dist = patient_params.get('k_distribution', 2.0)  # Distribution rate (day^-1)
        k_influx = patient_params.get('k_influx', 5.0)      # Influx into cells (day^-1)
        k_efflux_base = patient_params.get('k_efflux_base', 0.5)  # Base efflux rate (day^-1)
    else:  # paclitaxel or combination
        drug_model = PaclitaxelPKPD()
        k_dist = patient_params.get('k_distribution', 1.0)
        k_influx = patient_params.get('k_influx', 5.0)
        k_efflux_base = patient_params.get('k_efflux_base', 0.5)
    
    # ABC transporter parameters - affect intracellular efflux
    Vmax = patient_params['vmax_abc'] * ABC_expr  # Scale by expression level
    Km_abc = patient_params['km_abc']
    
    # Systemic clearance (convert from hr^-1 to day^-1)
    k_clear = patient_params['drug_clearance_rate'] * 24
    
    # --- ABC-mediated efflux (Michaelis-Menten) ---
    # Affects intracellular drug concentration
    abc_efflux_factor = (Vmax * D_intra) / (Km_abc + D_intra)
    
    # --- Drug kill terms (Hill equation) ---
    # Use intracellular concentration for cytotoxicity
    kill_S = (D_intra**hill_coeff) / (EC50_S**hill_coeff + D_intra**hill_coeff)
    kill_R = (D_intra**hill_coeff) / (EC50_R**hill_coeff + D_intra**hill_coeff)
    
    # --- Phenotypic switching (drug-induced) ---
    # Use intracellular concentration for drug pressure
    switch_S_to_R = mu_rate * sigma * S * (1 + 0.5 * D_intra / (EC50_S + D_intra))
    switch_R_to_S = 0.5 * mu_rate * sigma * R
    
    # --- Logistic growth with competition ---
    # Standard tumor growth model with carrying capacity
    N_total = S + R
    growth_factor = 1 - (N_total / K)
    
    # --- dS/dt: Sensitive cell population ---
    # High epigenetic instability increases drug sensitivity
    sigma_sensitivity = 1.0 / (1.0 + sigma * 0.1)
    dSdt = (
        r_S * S * growth_factor  # Logistic growth
        - d * S  # Baseline death
        - kill_S * patient_params['max_kill_rate'] * S / sigma_sensitivity  # Drug kill
        - switch_S_to_R  # Transition to resistant
        + switch_R_to_S  # Reversion from resistant
    )
    
    # --- dR/dt: Resistant cell population ---
    # ABC expression reduces intracellular drug levels
    abc_growth_bonus = 1.0 + (ABC_expr**1.5 * 0.50)
    
    dRdt = (
        r_R * R * growth_factor * abc_growth_bonus  # Growth with ABC bonus
        - d * R  # Baseline death
        - kill_R * patient_params['max_kill_rate'] * R  # Kill reduced by ABC efflux
        + switch_S_to_R  # Gain from sensitive
        - switch_R_to_S  # Loss to sensitive
    )
    
    # --- PK/PD System: Three-compartment model ---
    # dD_plasma/dt: Plasma compartment
    drug_input = drug_schedule_func(t)  # Pulsatile dosing (μM/day)
    dD_plasma_dt = (
        drug_input  # Dosing input
        - k_clear * D_plasma  # Systemic clearance
        - k_dist * (D_plasma - D_tumor)  # Distribution to tumor
    )
    
    # dD_tumor/dt: Tumor extracellular compartment
    dD_tumor_dt = (
        k_dist * (D_plasma - D_tumor)  # Distribution from plasma
        - k_influx * D_tumor  # Influx into cells
        + k_efflux_base * D_intra  # Base efflux from cells
        + ABC_expr * abc_efflux_factor  # ABC-enhanced efflux
    )
    
    # dD_intra/dt: Intracellular compartment
    dD_intra_dt = (
        k_influx * D_tumor  # Influx from extracellular
        - k_efflux_base * D_intra  # Base efflux out
        - ABC_expr * abc_efflux_factor  # ABC-enhanced efflux
    )
    
    # --- dABC/dt: ABC transporter expression ---
    # Induced by intracellular drug exposure
    abc_induction_rate = patient_params.get('abc_induction_rate', 0.1)
    abc_decay_rate = patient_params.get('abc_decay_rate', 0.05)
    abc_max = patient_params.get('abc_max_expression', 5.0)
    
    induction = abc_induction_rate * D_intra * (abc_max - ABC_expr) / (2.0 + D_intra)
    decay = abc_decay_rate * ABC_expr
    
    dABCdt = induction - decay
    
    # --- dE/dt: Epigenetic instability score ---
    # Accumulates under intracellular drug pressure
    sigma_baseline = patient_params['baseline_sigma']
    sigma_max = patient_params.get('sigma_max', 2.5)
    epigenetic_induction = 0.05 * D_intra * (sigma_max - E_score) / (1.0 + D_intra)
    epigenetic_decay = 0.02 * (E_score - sigma_baseline)
    
    dEscore_dt = epigenetic_induction - epigenetic_decay
    
    return [dSdt, dRdt, dD_plasma_dt, dD_tumor_dt, dD_intra_dt, dABCdt, dEscore_dt]

class PatientProfile:
    """Encapsulates all patient-specific biological parameters"""
    
    def __init__(self, stage, histology, residual_burden, 
                 baseline_plasticity, abc_expression):
        self.stage = stage
        self.histology = histology
        self.residual_burden = residual_burden
        self.baseline_plasticity = baseline_plasticity
        self.abc_expression = abc_expression
        
    def get_growth_rate(self, biomarker_risk_score: float = 0.5):
        """
        Returns histology, stage, ABC, and biomarker-adjusted growth rate
        
        Calibrated to give realistic recurrence times:
        - Low risk (score ~0.2): median DFS ~24-30 months
        - Average risk (score ~0.5): median DFS ~18-22 months  
        - High risk (score ~0.8+): median DFS ~12-15 months
        
        Args:
            biomarker_risk_score: 0-1 score from biomarkers (higher = more aggressive)
        """
        from utils.literature_params import GROWTH_RATES, STAGE_GROWTH_MULTIPLIERS
        
        # Select base growth rate based on combined risk (ABC + biomarkers)
        combined_risk = 0.4 * self.abc_expression + 0.6 * biomarker_risk_score
        
        if self.histology in GROWTH_RATES:
            rates = GROWTH_RATES[self.histology]
            if combined_risk < 0.3:
                base_rate = rates.get('very_low', 0.010)
            elif combined_risk < 0.5:
                base_rate = rates.get('low', 0.015)
            elif combined_risk < 0.7:
                base_rate = rates.get('medium', 0.022)
            elif combined_risk < 0.85:
                base_rate = rates.get('high', 0.030)
            else:
                base_rate = rates.get('very_high', 0.038)
        else:
            base_rate = 0.022  # Default medium
        
        # Apply stage-specific multiplier
        stage_mult = STAGE_GROWTH_MULTIPLIERS.get(self.stage, 1.0)
        
        return base_rate * stage_mult
    
    def get_carrying_capacity(self):
        """Returns stage-specific carrying capacity"""
        from utils.literature_params import CARRYING_CAPACITY
        
        stage_key = f"stage_{self.stage.replace('I', 'I')[:3]}"  # Normalize stage
        return CARRYING_CAPACITY.get(stage_key, 5e9)
    
    def to_params_dict(self, biomarker_risk_score: float = 0.5):
        """
        Convert patient profile to parameter dictionary for ODE solver
        
        Args:
            biomarker_risk_score: 0-1 score (higher = more aggressive, worse drug response)
        """
        from utils.literature_params import CARBOPLATIN, ABC_TRANSPORTERS, EPIGENETIC_PARAMS
        
        # Risk-adjusted growth rate
        growth_rate = self.get_growth_rate(biomarker_risk_score)
        
        # Risk-adjusted drug efficacy:
        # High risk → tumor is more resistant → higher EC50, lower max kill
        # EC50 range: 0.5 (sensitive) to 2.0 (resistant)
        ec50_sensitive_adj = CARBOPLATIN['ec50_sensitive'] * (0.8 + 0.8 * biomarker_risk_score)
        ec50_resistant_adj = CARBOPLATIN['ec50_resistant'] * (0.9 + 0.4 * biomarker_risk_score)
        
        # Max kill rate: 20% for low risk, 10% for high risk
        max_kill_adj = 0.20 - 0.10 * biomarker_risk_score
        
        return {
            'growth_rate_sensitive': growth_rate,
            'growth_rate_resistant': growth_rate * 1.05,  # Resistant cells slightly faster
            'carrying_capacity': self.get_carrying_capacity(),
            'death_rate': 0.005,  # baseline apoptosis ~0.5% per day
            'plasticity_rate': self.baseline_plasticity,
            'baseline_sigma': EPIGENETIC_PARAMS['default_sigma'],
            'sigma_max': 2.5,
            'ec50_sensitive': ec50_sensitive_adj,
            'ec50_resistant': ec50_resistant_adj,
            'hill_coefficient': 2.0,
            'max_kill_rate': max_kill_adj,
            'drug_clearance_rate': CARBOPLATIN['clearance_rate'],
            'vmax_abc': ABC_TRANSPORTERS['Vmax_abcc1'],
            'km_abc': ABC_TRANSPORTERS['Km_carboplatin'],
            'abc_induction_rate': ABC_TRANSPORTERS['induction_per_cycle'],
            'abc_decay_rate': 0.03,
            'abc_max_expression': 4.0,
            # PK parameters
            'drug_type': 'carboplatin',
            'k_distribution': 1.5,
            'k_influx': 4.0,
            'k_efflux_base': 0.3,
            # Store risk score for reference
            'biomarker_risk_score': biomarker_risk_score,
        }
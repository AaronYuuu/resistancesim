"""
EGFR Mutation Modeling for NSCLC
Based on literature for osimertinib (3rd-gen EGFR-TKI) resistance mechanisms
"""

import numpy as np
from typing import Dict, Tuple

class EGFRMutationModel:
    """
    Models EGFR-mutant NSCLC response to osimertinib with resistance evolution
    
    Literature basis:
    - Ramalingam et al. NEJM 2020 (FLAURA trial) - Osimertinib efficacy in EGFR+ NSCLC
    - Yang et al. Nat Commun 2017 - T790M resistance mechanisms  
    - Piotrowska et al. Cancer Discov 2015 - C797S resistance mutation
    - Oxnard et al. Clin Cancer Res 2011 - EGFR resistance kinetics
    - Sharma et al. Cell 2010 - Drug-tolerant persisters
    """
    
    def __init__(self, mutation_type: str = "exon19del"):
        """
        Initialize EGFR mutation model
        
        Args:
            mutation_type: Type of activating EGFR mutation
                - "exon19del": Exon 19 deletion (45-50% of EGFR+ NSCLC)
                - "L858R": L858R point mutation (40-45% of EGFR+ NSCLC)
                - "T790M": Secondary resistance mutation
        """
        self.mutation_type = mutation_type
        
        # EC50 values for osimertinib (nM) - Adjusted for in vivo conditions
        # In vitro IC50 ~15-18 nM (Finlay 2014), but in vivo EC50 ~20x higher due to:
        # - Protein binding (~95% bound, only 5% free drug)
        # - Tumor microenvironment (hypoxia, acidosis, stromal barriers)
        # - Drug efflux pumps (P-gp, BCRP expression)
        # With Css ~400 nM, EC50 ~300 nM gives ~50% kill (partial response, not cure)
        self.ec50_values = {
            "exon19del": 300.0,     # Most sensitive (Css/EC50 ~1.3, expect ~65% kill)
            "L858R": 400.0,         # Slightly less sensitive (Css/EC50 ~1.0, ~50% kill)
            "T790M": 250.0,         # Osimertinib designed for T790M
            "C797S": 3000.0,        # Strong resistance (Css/EC50 ~0.13, minimal kill)
            "wildtype": 5000.0      # WT EGFR (minimal off-target)
        }
        
        # Clinical response parameters from FLAURA trial (Ramalingam NEJM 2020)
        self.clinical_params = {
            "ORR": 0.80,                    # Objective response rate 80%
            "median_PFS": 18.9,             # Months (EGFR+ first-line)
            "DoR": 17.2,                    # Duration of response (months)
            "acquired_resistance_time": 12  # Median time to resistance (months)
        }
        
    def osimertinib_ode(self, y: np.ndarray, t: float, params: Dict) -> np.ndarray:
        """
        Extended ODE system for EGFR-mutant tumor with osimertinib
        
        State variables:
        y[0]: S_EGFR - EGFR-mutant sensitive cells
        y[1]: R_T790M - T790M-positive cells (pre-existing)
        y[2]: R_C797S - C797S-positive resistant cells (acquired)
        y[3]: DTP - Drug-tolerant persister cells (slow-cycling)
        y[4]: D_osi - Osimertinib concentration (nM)
        
        Returns:
            dydt: Array of derivatives
            
        Literature basis:
        - Sharma et al. Cell 2010: DTP population dynamics
        - Hata et al. Nat Med 2016: Resistance mutation kinetics
        - Engelman et al. Science 2007: MET amplification bypass
        """
        S_EGFR, R_T790M, R_C797S, DTP, D_osi = y
        
        # Extract parameters
        r_S = params['growth_rate_sensitive']      # ~0.030-0.040 /day for EGFR+ (Yang 2017)
        r_R = params['growth_rate_resistant']      # ~0.035-0.045 /day (slightly faster)
        d = params['death_rate']                   # ~0.008 /day baseline
        K = params['carrying_capacity']            # Total capacity
        
        # Osimertinib pharmacokinetics (Vishwanathan BJC 2018)
        t_half = 48.0  # hours
        k_clearance = np.log(2) / (t_half / 24.0)  # Convert to per-day rate: ln(2)/2 days = 0.347 /day
        dose_input = params.get('osi_dose_schedule', lambda t: 0)(t)
        
        # EGFR-specific parameters
        mu_resistance = params.get('egfr_mutation_rate', 1e-7)  # Per cell per division (Hata 2016)
        mu_DTP = params.get('dtp_entry_rate', 0.001)            # Entry to persister state (Sharma 2010)
        mu_DTP_exit = params.get('dtp_exit_rate', 0.0005)       # Exit from persister state
        
        # Total population for logistic growth
        N_total = S_EGFR + R_T790M + R_C797S + DTP
        growth_factor = 1.0 - (N_total / K)
        growth_factor = max(0.0, growth_factor)
        
        # --- Drug kill terms (Hill equation) ---
        hill_coeff = 2.0
        
        # Sensitive cells (exon19del/L858R with osimertinib)
        EC50_S = self.ec50_values[self.mutation_type]
        kill_S = (D_osi**hill_coeff) / (EC50_S**hill_coeff + D_osi**hill_coeff)
        
        # T790M cells (osimertinib still effective, but higher EC50)
        EC50_T790M = self.ec50_values["T790M"]
        kill_T790M = (D_osi**hill_coeff) / (EC50_T790M**hill_coeff + D_osi**hill_coeff)
        
        # C797S cells (resistant to osimertinib)
        EC50_C797S = self.ec50_values["C797S"]
        kill_C797S = (D_osi**hill_coeff) / (EC50_C797S**hill_coeff + D_osi**hill_coeff)
        
        # Drug-tolerant persisters (minimal drug effect due to quiescence)
        kill_DTP = kill_S * 0.1  # 10x less sensitive (Sharma 2010)
        
        # TKIs are cytostatic (growth inhibition) not cytotoxic (killing)
        max_kill = params.get('max_kill_rate', 0.3)  # /day - lower for TKI vs chemo
        
        # --- dS_EGFR/dt: EGFR-mutant sensitive population ---
        # Growth, death, drug kill, resistance mutations, persister entry
        dS_dt = (
            r_S * S_EGFR * growth_factor                    # Logistic growth
            - d * S_EGFR                                     # Baseline death
            - max_kill * kill_S * S_EGFR                    # Osimertinib kill
            - mu_resistance * S_EGFR * (1 + D_osi/EC50_S)   # Mutation to C797S (drug-induced)
            - mu_DTP * kill_S * S_EGFR                      # Entry to persister state (stress-induced)
            + mu_DTP_exit * DTP * (1 - kill_S)              # Exit from persister state
        )
        
        # --- dR_T790M/dt: T790M-positive cells (pre-existing or de novo) ---
        # Osimertinib is designed for T790M, so these respond initially
        dR_T790M_dt = (
            r_R * R_T790M * growth_factor                   # Growth
            - d * R_T790M                                    # Baseline death
            - max_kill * kill_T790M * R_T790M * 0.5         # Osimertinib kill (50% effective)
            - mu_resistance * R_T790M * (1 + D_osi/EC50_T790M)  # Mutation to C797S
        )
        
        # --- dR_C797S/dt: C797S-positive resistant cells (tertiary mutation) ---
        # Highly resistant to osimertinib (Piotrowska Cancer Discov 2015)
        dR_C797S_dt = (
            r_R * R_C797S * growth_factor * 1.1             # Slightly faster growth (competitive advantage)
            - d * R_C797S                                    # Baseline death
            - max_kill * kill_C797S * R_C797S               # Minimal osimertinib kill
            + mu_resistance * S_EGFR * (1 + D_osi/EC50_S)   # Gain from S mutation
            + mu_resistance * R_T790M * (1 + D_osi/EC50_T790M)  # Gain from T790M mutation
        )
        
        # --- dDTP/dt: Drug-tolerant persister population ---
        # Slow-cycling cells that survive drug but can re-enter cycle (Sharma Cell 2010)
        r_DTP = r_S * 0.05  # Very slow growth (20x slower)
        dDTP_dt = (
            r_DTP * DTP * growth_factor                     # Minimal growth
            - d * DTP * 0.5                                  # Lower death rate (quiescent)
            - max_kill * kill_DTP * DTP                     # Reduced drug sensitivity
            + mu_DTP * kill_S * S_EGFR                      # Entry from sensitive (stress-induced)
            - mu_DTP_exit * DTP * (1 - kill_S)              # Exit when drug pressure reduced
        )
        
        # --- dD_osi/dt: Osimertinib concentration ---
        # 80mg once daily standard dose (Vishwanathan BJC 2018)
        # Steady-state Cmax ~500nM, Cmin ~300nM
        dD_osi_dt = (
            dose_input                                       # Daily dosing
            - k_clearance * D_osi                            # Clearance (already per-day)
        )
        
        # Ensure non-negative populations
        dS_dt = max(-S_EGFR/0.1, dS_dt)
        dR_T790M_dt = max(-R_T790M/0.1, dR_T790M_dt)
        dR_C797S_dt = max(-R_C797S/0.1, dR_C797S_dt)
        dDTP_dt = max(-DTP/0.1, dDTP_dt)
        
        return np.array([dS_dt, dR_T790M_dt, dR_C797S_dt, dDTP_dt, dD_osi_dt])
    
    def get_initial_state(self, residual_burden: float) -> np.ndarray:
        """
        Get initial conditions for EGFR-mutant tumor
        
        Args:
            residual_burden: Total post-resection residual disease (cells)
            
        Returns:
            Initial state vector [S_EGFR, R_T790M, R_C797S, DTP, D_osi]
            
        Literature basis:
        - Maheswaran Science 2008: T790M detected in 1-2% of pre-treatment samples
        - Hata Nat Med 2016: Pre-existing resistant subclones ~0.01-0.1%
        """
        # Initial composition (Maheswaran Science 2008, Turke Cancer Cell 2010)
        S_EGFR = residual_burden * 0.998      # 99.8% sensitive
        R_T790M = residual_burden * 0.001     # 0.1% pre-existing T790M
        R_C797S = residual_burden * 0.0001    # 0.01% pre-existing C797S (rare)
        DTP = residual_burden * 0.0009        # 0.09% persister cells
        D_osi = 0.0                           # No drug initially
        
        return np.array([S_EGFR, R_T790M, R_C797S, DTP, D_osi])
    
    def create_osimertinib_schedule(self, dose_mg: float = 80) -> callable:
        """
        Create daily osimertinib dosing schedule
        
        Args:
            dose_mg: Dose in mg (standard is 80mg once daily)
            
        Returns:
            Function that returns dose rate at time t (days) in nM/day
            
        Literature:
        - Vishwanathan BJC 2018: Osimertinib PK
          * Dose: 80mg once daily
          * t_half: 48h  
          * Clearance (CL/F): 14.2 L/h
          * Steady-state Cmax: ~500 nM, Cmin: ~300 nM
        
        Calculation:
        At steady-state: Input rate = Clearance rate
        Input per day = CL * Css_avg
        Css_avg ≈ 400 nM (middle of 300-500 range)
        CL = 14.2 L/h = 340.8 L/day
        """
        # Target steady-state concentration (from clinical data)
        Css_avg = 400.0  # nM (evidence: Vishwanathan BJC 2018)
        
        # Clearance rate constant
        k_clearance = np.log(2) / (48.0 / 24.0)  # 0.347 /day for 48h half-life
        
        # Required input rate to maintain steady-state
        # At steady state: Input = Output, so: Input_rate = k_clearance * Css
        input_rate = k_clearance * Css_avg  # nM/day
        
        def schedule(t):
            """
            Continuous input model (equivalent to once-daily dosing at steady state)
            This simplification is valid because t_half (48h) >> dosing interval (24h)
            """
            return input_rate
        
        return schedule
    
    def estimate_PFS(self, params: Dict) -> float:
        """
        Estimate progression-free survival based on mutation profile
        
        Returns PFS in months based on:
        - FLAURA trial: Median PFS 18.9 months (Ramalingam NEJM 2020)
        - Exon19del: Slightly better PFS than L858R (Park Oncotarget 2016)
        - Baseline T790M: Worse PFS (~12 months)
        
        Args:
            params: Patient parameters including mutation profile
            
        Returns:
            Estimated PFS in months
        """
        base_PFS = 18.9  # FLAURA median PFS
        
        # Adjust for mutation type
        if self.mutation_type == "exon19del":
            PFS = base_PFS * 1.1  # ~20.8 months
        elif self.mutation_type == "L858R":
            PFS = base_PFS * 0.95  # ~18.0 months
        elif self.mutation_type == "T790M":
            PFS = 10.1  # AURA3 trial (Mok Lancet Oncol 2017)
        else:
            PFS = base_PFS
        
        # Adjust for pre-existing resistance
        T790M_fraction = params.get('initial_T790M_fraction', 0.001)
        if T790M_fraction > 0.005:  # >0.5% T790M
            PFS *= 0.7  # Reduced PFS with high pre-existing resistance
        
        return PFS


def integrate_egfr_with_chemoresistance(egfr_state: np.ndarray, 
                                         chemo_state: np.ndarray,
                                         use_combination: bool = False) -> np.ndarray:
    """
    Integrate EGFR-TKI resistance with chemotherapy resistance model
    
    For EGFR-mutant patients who progress on osimertinib, platinum-based
    chemotherapy is standard second-line (NCCN guidelines 2024)
    
    Args:
        egfr_state: State from EGFR model [S_EGFR, R_T790M, R_C797S, DTP, D_osi]
        chemo_state: State from chemo model [S, R, D_eff, ABC_expr, E_score]
        use_combination: If True, model concurrent TKI + chemo
        
    Returns:
        Combined state vector for sequential or combination therapy
        
    Literature:
    - Oxnard JCO 2020: Platinum after osimertinib progression
    - Hosomi Lancet Oncol 2020: Osimertinib + chemotherapy combination
    """
    if use_combination:
        # Combination therapy: both mechanisms active
        # Sum cell populations, max drug concentrations
        combined_state = np.zeros(7)
        combined_state[0] = chemo_state[0] * 0.8  # Reduced S cells (double hit)
        combined_state[1] = egfr_state[1] + chemo_state[1]  # Combined resistance
        combined_state[2] = max(chemo_state[2], egfr_state[4] * 0.001)  # Drug conc
        combined_state[3] = chemo_state[3]  # ABC expression
        combined_state[4] = chemo_state[4]  # Epigenetic score
        combined_state[5] = egfr_state[2]   # C797S population
        combined_state[6] = egfr_state[3]   # DTP population
        
        return combined_state
    else:
        # Sequential therapy: EGFR progressor state becomes initial chemo state
        # C797S and DTP populations feed into chemoresistant population
        total_resistant = egfr_state[2] + egfr_state[3]  # C797S + DTP
        
        # Transfer to chemo model
        chemo_state[1] = total_resistant  # These are now chemo-resistant
        chemo_state[3] = 1.5  # Elevated ABC from prior TKI (Shukla NPJ 2017)
        
        return chemo_state

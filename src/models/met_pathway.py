"""
MET Pathway Bypass Module for EGFR-TKI Resistance
Models HGF/MET-driven resistance as an alternative signaling axis
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

from src.models.mutations import EGFRMutationModel

class METPathwayModule:
    """
    MET amplification bypass resistance module.
    
    Literature basis:
    - Engelman et al. Science 2007: MET amplification drives EGFR-TKI resistance
    - Bean et al. PNAS 2007: HGF rescues EGFR+ cells from gefitinib
    - Oxnard et al. JCO 2012: Prevalence 5-10% of EGFR-TKI resistance
    - Wu et al. Cancer Cell 2016: HGF from CAFs activates MET
    - Sennino et al. Cancer Res 2012: Dual EGFR+MET inhibition
    """
    
    def __init__(self, baseline_met_copies: int = 2):
        """
        Initialize MET pathway module
        
        Args:
            baseline_met_copies: MET gene copy number (normal diploid = 2)
        """
        self.baseline_met_copies = baseline_met_copies
        self.current_met_copies = baseline_met_copies
        
        # MET signaling parameters (Engelman 2007, Wu 2016)
        self.met_kinase_activity = 0.1  # Arbitrary units
        self.hgf_sensitivity = 1.0      # EC50 scaling factor
        
    def get_parameters(self) -> Dict[str, float]:
        """Return module-specific parameters for ODE integration"""
        return {
            'met_signal_strength': self.current_met_copies * self.met_kinase_activity,
            'hgf_production_rate': 0.05,        # ng/mL/day from CAFs (Wu 2016)
            'met_activation_threshold': 1.5,    # Copies needed for significant bypass
            'met_drug_resistance_factor': 5.0,  # 5x reduction in EGFR-TKI kill rate
        }
    
    def met_ode_term(self, tumor_populations: Dict[str, float], params: Dict) -> Dict[str, float]:
        """
        Calculate MET pathway contribution to tumor dynamics
        
        Args:
            tumor_populations: Dict with 'S' (sensitive) and 'R' (resistant) cell counts
            params: Model parameters including MET status
            
        Returns:
            Dictionary of dS/dt and dR/dt contributions from MET pathway
        """
        # Check if MET amplification active
        met_copies = params.get('MET_amplification_copies', self.current_met_copies)
        is_met_active = met_copies > params.get('met_activation_threshold', 1.5)
        
        if not is_met_active:
            return {'dS_dt_met': 0.0, 'dR_dt_met': 0.0}
        
        # HGF concentration (from stroma, microenvironment)
        hgf_level = params.get('serum_hgf_pg_ml', 1.0)  # Use actual patient data if available
        
        # MET signaling strength (Bean PNAS 2007)
        # HGF activates MET → ERBB3/PI3K bypass → survival even with EGFR blocked
        met_signal = met_copies * hgf_level * self.met_kinase_activity
        
        # Resistance contribution (Engelman Science 2007)
        # MET+ cells survive EGFR inhibition and proliferate via alternative pathway
        growth_boost = met_signal * params.get('met_growth_boost', 0.2)
        
        # Reduce EGFR-TKI kill rate (Wu Cancer Cell 2016)
        resistance_factor = params.get('met_drug_resistance_factor', 5.0)
        
        # Calculate contributions to tumor dynamics
        # S cells become phenotypically resistant via MET activation
        met_conversion_rate = met_signal * 0.01  # % per day converting to MET-dependent state
        
        dS_dt_met = (
            -met_conversion_rate * tumor_populations['S']  # Loss of EGFR-dependent cells
            - (resistance_factor - 1) * params.get('max_kill_rate', 0.3) * tumor_populations['S']  # Reduced TKI kill
        )
        
        dR_dt_met = (
            +met_conversion_rate * tumor_populations['S']  # Gain of MET-dependent resistant cells
            + growth_boost * tumor_populations['R']       # Enhanced growth of resistant population
        )
        
        return {'dS_dt_met': dS_dt_met, 'dR_dt_met': dR_dt_met}
    
    def integrate_with_tumor_ode(self, 
                                 tumor_dydt: np.ndarray, 
                                 y: np.ndarray, 
                                 params: Dict,
                                 egfr_model: 'EGFRMutationModel' = None) -> np.ndarray:
        """
        Integrate MET pathway terms with main tumor ODE
        
        Args:
            tumor_dydt: Current tumor ODE derivatives [dS_dt, dR_dt, ...]
            y: Current state vector [S, R, ...]
            params: Full parameter dictionary
            egfr_model: Optional EGFR model for combination effects
            
        Returns:
            Modified derivative vector with MET contributions
        """
        # Extract tumor populations
        tumor_pops = {'S': y[0], 'R': y[1]}
        
        # Get MET contributions
        met_terms = self.met_ode_term(tumor_pops, params)
        
        # Add to existing derivatives
        tumor_dydt[0] += met_terms['dS_dt_met']
        tumor_dydt[1] += met_terms['dR_dt_met']
        
        # Triple therapy scenario (EGFR + MET inhibitor)
        if egfr_model is not None and params.get('met_inhibitor_active', False):
            # MET inhibitor reduces conversion rate and resistance factor
            met_inhibitor_conc = params.get('met_inhibitor_conc', 0.0)
            met_ec50 = 100.0  # nM (savolitinib-like inhibitor)
            
            inhibition = met_inhibitor_conc / (met_inhibitor_conc + met_ec50)
            
            # Revert MET-driven effects
            tumor_dydt[0] += (1 - inhibition) * abs(met_terms['dS_dt_met'])
            tumor_dydt[1] += (1 - inhibition) * met_terms['dR_dt_met'] * 0.5
        
        return tumor_dydt
    
    def predict_met_resistance_probability(self, 
                                         ctdna_data: Dict,
                                         hgf_level: float) -> float:
        """
        Predict probability of MET-driven resistance from ctDNA and HGF data
        
        Args:
            ctdna_data: ctDNA measurements with MET amplification status
            hgf_level: Serum HGF concentration (pg/mL)
            
        Returns:
            Probability (0-1) of MET-mediated resistance
        """
        # Base probability from ctDNA
        met_amp_vaf = ctdna_data.get('met_amp_vaf', 0.0)
        met_copies = ctdna_data.get('met_copies', 2)
        
        # HGF contribution (Bean PNAS 2007)
        # HGF > 2 ng/mL significantly associated with resistance
        hgf_contribution = min(hgf_level / 2000.0, 1.0)  # Normalize to 2 ng/mL
        
        # Combined probability
        prob_met_ctdna = 1.0 if met_amp_vaf > 0.01 else 0.0  # Binary if detected
        prob_met_hgf = 0.3 if hgf_contribution > 0.5 else 0.1  # 30% if high HGF
        
        # Weighted average
        return max(prob_met_ctdna, prob_met_hgf)
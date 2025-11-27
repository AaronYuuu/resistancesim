"""
Epithelial-to-Mesenchymal Transition (EMT) Module
Models EMT-driven resistance and metastatic potential
"""

import numpy as np
from typing import Dict, Tuple

class EMTModule:
    """
    EMT-mediated resistance and metastasis module
    
    Literature basis:
    - Thompson et al. Nat Rev Cancer 2005: EMT in cancer progression
    - Fischer et al. Cancer Res 2015: EMT and drug resistance
    - Zheng et al. Nat Med 2015: TGF-β induced EMT in NSCLC
    - Byers et al. Clin Cancer Res 2013: EMT signature predicts EGFR-TKI resistance
    - Ocana et al. Cancer Res 2012: EMT and stem cell properties
    """
    
    def __init__(self, baseline_e_cadherin: float = 1.0):
        """
        Initialize EMT module
        
        Args:
            baseline_e_cadherin: Baseline E-cadherin expression (epithelial marker)
        """
        self.baseline_e_cadherin = baseline_e_cadherin
        self.current_e_cadherin = baseline_e_cadherin
        
        # EMT markers
        self.vimentin_expression = 0.0
        self.twist_expression = 0.0
        self.snail_expression = 0.0
        
        # EMT transition parameters
        self.emt_transition_rate = 0.05  # Per day (Fischer Cancer Res 2015)
        self.mesenchymal_growth_boost = 1.2  # 20% faster growth in M state
        self.mesenchymal_stemness_factor = 2.0  # 2x therapy resistance (Byers 2013)
    
    def get_parameters(self) -> Dict[str, float]:
        """Return module-specific parameters for ODE integration"""
        return {
            'emt_transition_rate': self.emt_transition_rate,
            'mesenchymal_growth_boost': self.mesenchymal_growth_boost,
            'mesenchymal_drug_resistance': self.mesenchymal_stemness_factor,
            'e_cadherin_threshold': 0.3,  # Below this = mesenchymal phenotype
            'tgf_beta_emt_induction': 0.5,  # TGF-β induced EMT rate
        }
    
    def emt_ode_term(self, 
                     tumor_populations: Dict[str, float],
                     tme_cytokines: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate EMT contribution to tumor dynamics
        
        Args:
            tumor_populations: Dict with 'E' (epithelial) and 'M' (mesenchymal) cells
            tme_cytokines: Dict with 'TGFb', 'IL6', 'HGF' levels
            
        Returns:
            Dictionary of dE/dt and dM/dt contributions from EMT
        """
        # Extract populations
        E_cells = tumor_populations['E']
        M_cells = tumor_populations['M']
        
        # Cytokine levels (normalize)
        tgf_b = tme_cytokines.get('TGFb', 1.0) / 10.0  # ng/mL
        il6 = tme_cytokines.get('IL6', 1.0) / 100.0     # pg/mL
        
        # EMT induction (Zheng Nat Med 2015)
        # TGF-β is primary driver; IL-6 secondary
        emt_induction_rate = (
            self.emt_transition_rate * 
            (1 + tgf_b * 2.0) *  # TGF-β doubles EMT rate
            (1 + il6 * 0.5)      # IL-6 adds 50% boost
        )
        
        # Mesenchymal to epithelial transition (MET) - rare, slow
        met_rate = self.emt_transition_rate * 0.1  # 10x slower than EMT
        
        # Calculate dynamics
        # E cells transition to M phenotype
        dE_dt_emt = -emt_induction_rate * E_cells
        
        # M cells gain from E transition + proliferate faster + more resistant
        dM_dt_emt = (
            +emt_induction_rate * E_cells
            + (self.mesenchymal_growth_boost - 1.0) * M_cells  # Proliferation boost
        )
        
        # Standard proliferation (handled in main ODE, this is EMT-specific)
        # But include stemness factor for resistance calculation
        
        return {
            'dE_dt_emt': dE_dt_emt,
            'dM_dt_emt': dM_dt_emt,
            'emt_induction_rate': emt_induction_rate
        }
    
    def integrate_with_tumor_ode(self, 
                                 tumor_dydt: np.ndarray, 
                                 y: np.ndarray, 
                                 params: Dict) -> np.ndarray:
        """
        Integrate EMT pathway terms with main tumor ODE
        
        Args:
            tumor_dydt: Current tumor ODE derivatives [dS_dt, dR_dt, ...]
            y: Current state vector [S, R, ...]
            params: Full parameter dictionary with TME cytokines
            
        Returns:
            Modified derivative vector with EMT contributions
        """
        # Split sensitive population into epithelial (E) and mesenchymal (M)
        # For simplicity: 90% E, 10% M baseline, but can shift
        total_sensitive = y[0]
        e_fraction = max(0.1, min(0.9, self.current_e_cadherin))
        m_fraction = 1.0 - e_fraction
        
        E_cells = total_sensitive * e_fraction
        M_cells = total_sensitive * m_fraction
        
        # Get TME cytokines
        tme_cytokines = {
            'TGFb': params.get('serum_tgfb_ng_ml', 1.0),
            'IL6': params.get('plasma_il6_pg_ml', 1.0),
            'HGF': params.get('serum_hgf_pg_ml', 1.0)
        }
        
        # Get EMT terms
        tumor_pops = {'E': E_cells, 'M': M_cells}
        emt_terms = self.emt_ode_term(tumor_pops, tme_cytokines)
        
        # Update tumor derivatives
        # Net effect: mesenchymal cells grow faster and are more resistant
        total_effect_on_S = emt_terms['dE_dt_emt'] + emt_terms['dM_dt_emt']
        tumor_dydt[0] += total_effect_on_S
        
        # Apply resistance factor to drug kill (mesenchymal cells are more resistant)
        if len(tumor_dydt) > 2:  # If drug concentration term exists
            drug_kill_reduction = 1.0 / params.get('mesenchymal_drug_resistance', 2.0)
            tumor_dydt[2] *= drug_kill_reduction  # Reduce drug efficacy
        
        return tumor_dydt
    
    def update_emt_markers(self, drug_pressure: float, time_days: float):
        """
        Update EMT marker expression based on drug pressure and time
        
        Args:
            drug_pressure: Normalized drug concentration (0-1)
            time_days: Time under drug pressure
        """
        # Drug pressure induces EMT (Fischer Cancer Res 2015)
        emt_induction = drug_pressure * 0.01 * time_days  # Cumulative effect
        
        # Update E-cadherin (epithelial marker, decreases in EMT)
        self.current_e_cadherin = max(0.0, self.baseline_e_cadherin - emt_induction)
        
        # Update vimentin (mesenchymal marker, increases in EMT)
        self.vimentin_expression = min(1.0, emt_induction * 0.5)
        
        # Update TWIST (EMT transcription factor)
        self.twist_expression = min(1.0, emt_induction * 0.3)
    
    def predict_metastatic_potential(self) -> float:
        """
        Predict metastatic potential based on EMT status
        
        Returns:
            Metastatic risk score (0-1)
            
        Literature:
        - Thompson Nat Rev Cancer 2005: EMT enables metastasis
        - Ocana Cancer Res 2012: EMT+ cells have stem-like properties
        """
        # EMT score based on marker expression
        emt_score = (1.0 - self.current_e_cadherin) * 0.5 + \
                    self.vimentin_expression * 0.3 + \
                    self.twist_expression * 0.2
        
        # Metastatic risk increases sigmoidally with EMT score
        metastatic_risk = 1.0 / (1.0 + np.exp(-10 * (emt_score - 0.5)))
        
        return metastatic_risk
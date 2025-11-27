import numpy as np

class CarboplatinPKPD:
    """
    One-compartment model for carboplatin pharmacokinetics
    """
    
    def __init__(self, auc_target=5, clearance_rate=0.173):
        self.auc_target = auc_target  # Area under curve
        self.clearance = clearance_rate  # t1/2 ~ 4 hours
        
    def concentration_time_profile(self, dose_mg, time_hours):
        """
        C(t) = C0 * exp(-k*t)
        
        C0 derived from dose and volume of distribution
        """
        Vd = 1.5  # L/kg, typical for carboplatin
        C0 = dose_mg / Vd
        
        concentration = C0 * np.exp(-self.clearance * time_hours)
        
        return concentration

class PaclitaxelPKPD:
    """
    Separate model for paclitaxel (different PK properties)
    """
    
    def __init__(self):
        self.half_life = 13.4  # hours
        self.clearance = np.log(2) / self.half_life
        
    # Similar structure to CarboplatinPKPD
    
class CombinationTherapy:
    """
    Handles combination regimens (carboplatin + paclitaxel)
    Calculates combined cytotoxic effect
    standard chemotherapy treatment for various cancers, including for NSCLC
    """
    
    def __init__(self, regimen="q21d"):
        self.carboplatin = CarboplatinPKPD()
        self.paclitaxel = PaclitaxelPKPD()
        self.schedule = self._parse_regimen(regimen)
        
    def get_total_drug_effect(self, t, dose_carboplatin, dose_paclitaxel):
        """
        Calculate combined effect using Bliss independence model
        
        E_total = E_carboplatin + E_paclitaxel - (E_carboplatin * E_paclitaxel)
        """
        # Get individual drug concentrations at time t
        C_carboplatin = self.carboplatin.concentration_time_profile(dose_carboplatin, t)
        C_paclitaxel = self.paclitaxel.concentration_time_profile(dose_paclitaxel, t)
        
        # Calculate individual effects using Hill equation
        E_carboplatin = self._hill_equation(C_carboplatin, ic50_carboplatin=1.2)
        E_paclitaxel = self._hill_equation(C_paclitaxel, ic50_paclitaxel=0.3)
        
        # Bliss independence for combination effect
        E_combined = E_carboplatin + E_paclitaxel - (E_carboplatin * E_paclitaxel)
        
        return E_combined
    
class osimertinib:
    def __init__(self):
        self.half_life = 48  # hours
        self.clearance = 14.3  # L/h, oral clearance (CL/F)
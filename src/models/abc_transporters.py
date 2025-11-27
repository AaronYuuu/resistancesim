class ABCMediatedEfflux:
    """
    Mechanistic model of P-gp (ABCB1) and ABCG2 drug efflux
    with chemotherapy-induced upregulation
    """
    
    def __init__(self, basal_abcc1=1.0, basal_abcg2=0.5):
        self.basal_abcc1 = basal_abcc1
        self.basal_abcg2 = basal_abcg2
        self.current_activity = {'ABCB1': basal_abcc1, 'ABCG2': basal_abcg2}
        
    def calculate_efflux_rate(self, intracellular_drug, chemotherapy_cycles):
        """
        Michaelis-Menten kinetics with cycle-dependent induction
        """
        # Chemotherapy induces ABC expression (ABC transporter upregulation)
        induction_factor = 1 + (chemotherapy_cycles * 0.15)  # Saturation at ~10 cycles
        
        Vmax_abcc1 = 8.5 * self.current_activity['ABCB1'] * induction_factor
        Vmax_abcg2 = 6.2 * self.current_activity['ABCG2'] * induction_factor
        
        # Combined efflux for carboplatin (substrate for both)
        Km_carboplatin = 2.3  # μM
        
        total_efflux = (Vmax_abcc1 * intracellular_drug / (Km_carboplatin + intracellular_drug)) + \
                       (Vmax_abcg2 * intracellular_drug / (Km_carboplatin + intracellular_drug))
        
        return total_efflux
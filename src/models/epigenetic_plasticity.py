import numpy as np

def softmax(x):
    """Numerically stable softmax."""
    x = np.asarray(x)
    x_max = np.max(x)
    e_x = np.exp(x - x_max)
    return e_x / e_x.sum()

class EpigeneticStateMachine:
    """
    Models heritable chemotherapy tolerance phenotypes (CTP) 
    via epigenetic noise and selection
    """
    
    def __init__(self, instability_sigma=0.5, heritability_h=0.8):
        self.sigma = instability_sigma  # Epigenetic noise parameter
        self.heritability = heritability_h
        self.ctp_states = np.array([0.1, 0.5, 1.0, 2.0])  # Low to High tolerance
        
    def inheritance_function(self, parental_ctp, drug_exposure):
        """
        Probabilistic inheritance of CTP under selection pressure
        Based on Lei et al. 2019 epigenetic plasticity models
        """
        # Drug pressure shifts distribution toward higher CTP
        selection_pressure = 1 + (drug_exposure * 2)
        
        # Epigenetic noise creates variability in inheritance
        noise = np.random.normal(0, self.sigma, size=len(self.ctp_states))
        
        # Weighted probability distribution
        inheritance_probs = softmax(parental_ctp * selection_pressure + noise)
        return inheritance_probs
"""
Auto-select ODE modules based on predicted resistance mechanism
"""
import sys
import os

import pandas as pd
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.ml.models.resistance_classifier import TMEGraphClassifier
from src.models.met_pathway import METPathwayModule
from src.models.emt_module import EMTModule

class ODESelector:
    """
    Maps predicted resistance mechanism to appropriate ODE module configuration
    """
    
    MODULE_CONFIGS = {
        'No Resistance': {
            'base_model': 'standard_tumor_population',
            'enable_met': False,
            'enable_emt': False,
            'param_overrides': {
                'mu': 0.05,  # Standard plasticity
                'ABC': 1.0
            }
        },
        'C797S': {
            'base_model': 'standard_tumor_population',
            'enable_met': False,
            'enable_emt': False,
            'param_overrides': {
                'mu': 0.2,  # High plasticity
                'ABC': 2.5,  # High efflux
                'r_R': 0.08  # Resistant cells grow faster
            }
        },
        'MET_amp': {
            'base_model': 'tumor_population_with_met_bypass',
            'enable_met': True,
            'enable_emt': False,
            'param_overrides': {
                'mu': 0.15,
                'ABC': 1.5,
                'met_signal_strength': 2.0  # Additional parameter
            }
        },
        'Loss_T790M': {
            'base_model': 'tumor_population_t790m_loss',
            'enable_met': False,
            'enable_emt': True,
            'param_overrides': {
                'mu': 0.3,  # Very high plasticity
                'ABC': 1.8,
                'emt_transition_rate': 0.1
            }
        },
        'Other': {
            'base_model': 'tumor_population_heterogeneous',
            'enable_met': True,
            'enable_emt': True,
            'param_overrides': {
                'mu': 0.25,
                'ABC': 2.0,
                'met_signal_strength': 1.0,
                'emt_transition_rate': 0.05
            }
        }
    }
    
    def __init__(self, classifier: 'TMEGraphClassifier'):
        self.classifier = classifier
        self.active_modules = {}
        
    def select_modules(self, patient_data: dict) -> dict:
        """
        Predict resistance mechanism and return appropriate ODE configuration
        """
        # Get prediction
        prediction = self.classifier.predict_from_patient_data(patient_data)
        mechanism = prediction['predicted_mechanism']
        
        # Get configuration
        config = self.MODULE_CONFIGS[mechanism].copy()
        config['prediction_confidence'] = prediction['confidence']
        config['prediction_uncertainty'] = prediction['uncertainty']
        config['all_probabilities'] = prediction['all_probabilities']
        
        # Dynamically load modules
        self._load_modules(config)
        
        return config
    
    def _load_modules(self, config: dict):
        """
        Dynamically import and instantiate ODE modules
        """
        # Clear previous modules
        self.active_modules.clear()
        
        # Load MET module if enabled
        if config['enable_met']:
            try:
                self.active_modules['met'] = METPathwayModule()
            except Exception as e:
                print(f"Warning: MET module not available: {e}")
        
        # Load EMT module if enabled
        if config['enable_emt']:
            try:
                self.active_modules['emt'] = EMTModule()
            except Exception as e:
                print(f"Warning: EMT module not available: {e}")
    
    def get_simulation_params(self, config: dict, patient_id: str) -> dict:
        """
        Merge module-specific parameters with patient-specific overrides
        """
        # Start with literature defaults
        from src.utils.literature_params import DEFAULT_PARAMS
        
        params = DEFAULT_PARAMS.copy()
        
        # Apply resistance-specific overrides
        params.update(config['param_overrides'])
        
        # Add module-specific parameters
        for module_name, module in self.active_modules.items():
            if hasattr(module, 'get_parameters'):
                params.update(module.get_parameters())
        
        # Store metadata
        params['selected_mechanism'] = config.get('predicted_mechanism')
        params['prediction_confidence'] = config.get('prediction_confidence', 0.0)
        
        return params

# Integration helper
def get_ode_config_for_patient(patient_id: str, ctdna_df: pd.DataFrame, tme_df: pd.DataFrame) -> dict:
    """
    High-level function for app integration
    """
    # Load GNN
    gnn = TMEGraphClassifier()
    checkpoint = torch.load('src/ml/checkpoints/tme_gnn_classifier.pth', map_location='cpu')
    gnn.load_state_dict(checkpoint['model_state_dict'])
    gnn.eval()
    
    # Build patient data dict
    patient_tme = tme_df[tme_df['patient_id'] == patient_id].iloc[0]
    patient_ctdna = ctdna_df[ctdna_df['patient_id'] == patient_id].iloc[0]
    
    patient_data = {
        'resistance_mechanism': patient_ctdna['resistance_mechanism'],
        'circulating_mdsc_per_ml': patient_tme['circulating_mdsc_per_ml'],
        'plasma_il10_pg_ml': patient_tme['plasma_il10_pg_ml'],
        'serum_hgf_pg_ml': patient_tme['serum_hgf_pg_ml'],
        'plasma_vegf_pg_ml': patient_tme['plasma_vegf_pg_ml']
    }
    
    # Select modules
    selector = ODESelector(gnn)
    config = selector.select_modules(patient_data)
    
    return config
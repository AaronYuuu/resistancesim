import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data, Dataset
from typing import List, Tuple, Optional

class TMEGraphBuilder:
    """
    Converts tabular TME data into graph structure for GNN training
    Nodes: Cell types (Tumor, Immune, Stromal subtypes)
    Edges: Known biological interactions
    """
    
    # Define cell types as nodes (12 node types)
    NODE_TYPES = {
        0: 'tumor_sensitive',
        1: 'tumor_resistant',
        2: 'cd8_tcell',
        3: 'm2_macrophage',
        4: 'mdsc',
        5: 'ca_fibroblast',
        6: 'endothelial_cell',
        7: 'nk_cell',
        8: 'dendritic_cell',
        9: 'b_cell',
        10: 'treg',
        11: 'neutrophil'
    }
    
    # Define biological interactions as edges (bidirectional)
    EDGE_DEFINITIONS = [
        # Tumor-Immune interactions
        (0, 2), (1, 2),  # Tumor -> CD8 T cells (recognition)
        (2, 0), (2, 1),  # CD8 -> Tumor (killing)
        (0, 3), (1, 3),  # Tumor -> M2 TAMs (recruitment)
        (3, 1),          # M2 TAMs -> Resistant tumor (support)
        (4, 2),          # MDSCs -> CD8 T cells (suppression)
        (5, 2),          # CAFs -> CD8 T cells (exclusion)
        (5, 3),          # CAFs -> M2 TAMs (activation)
        
        # Immune-Immune interactions
        (7, 0),          # NK -> Tumor (innate killing)
        (8, 2),          # DC -> CD8 T cells (priming)
        (9, 2),          # B -> CD8 T cells (antibody help)
        (10, 2),         # Treg -> CD8 T cells (suppression)
        (11, 3),         # Neutrophils -> M2 TAMs (polarization)
        (6, 4),          # Endothelial -> MDSCs (recruitment)
        
        # Autocrine loops
        (5, 5),          # CAF autocrine activation
        (3, 3)           # M2 TAM autocrine maintenance
    ]
    
    def __init__(self):
        # Create edge index tensor (2 x num_edges)
        edge_index = torch.tensor(self.EDGE_DEFINITIONS, dtype=torch.long).t()
        self.edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # Make bidirectional
        
    def build_graph_from_patient(self, 
                                 patient_id: str, 
                                 tissue_df: pd.DataFrame, 
                                 tme_blood_df: pd.DataFrame) -> Data:
        """
        Build graph for a single patient/timepoint
        
        Args:
            patient_id: Patient ID string
            tissue_df: Tissue biopsy data
            tme_blood_df: Blood TME factor data
            
        Returns:
            torch_geometric.data.Data object
        """
        # Get patient data
        tissue_data = tissue_df[tissue_df['patient_id'] == patient_id]
        blood_data = tme_blood_df[tme_blood_df['patient_id'] == patient_id]
        
        if tissue_data.empty or blood_data.empty:
            return self._create_dummy_graph()
        
        # Node features: [density, activation_score, spatial_metric]
        node_features = torch.zeros(len(self.NODE_TYPES), 3)
        
        # Populate node features from data
        # Tumor nodes (0,1)
        tumor_area = tissue_data['tumor_area_mm2'].iloc[0] if not tissue_data.empty else 100
        node_features[0, 0] = tumor_area * 0.6  # Sensitive fraction
        node_features[1, 0] = tumor_area * 0.4  # Resistant fraction
        
        # CD8 T cells (node 2)
        cd8_density = tissue_data['cd8_til_density_per_mm2'].iloc[0] if 'cd8_til_density_per_mm2' in tissue_data.columns else 150
        node_features[2, 0] = cd8_density
        node_features[2, 2] = tissue_data['cd8_tumor_distance_um'].iloc[0] / 100.0 if 'cd8_tumor_distance_um' in tissue_data.columns else 1.0
        
        # M2 TAMs (node 3)
        m2_density = tissue_data['m2_tam_density_per_mm2'].iloc[0] if 'm2_tam_density_per_mm2' in tissue_data.columns else 40
        node_features[3, 0] = m2_density
        node_features[3, 2] = tissue_data['immune_cell_distance_um'].iloc[0] / 100.0 if 'immune_cell_distance_um' in tissue_data.columns else 1.0
        
        # MDSCs (node 4)
        mdsc_blood = blood_data['circulating_mdsc_per_ml'].iloc[0] if not blood_data.empty else 50
        node_features[4, 0] = mdsc_blood / 1000.0  # Scale down
        
        # CAFs (node 5)
        caf_density = tissue_data['caf_density_per_mm2'].iloc[0] if 'caf_density_per_mm2' in tissue_data.columns else 60
        node_features[5, 0] = caf_density
        node_features[5, 1] = tissue_data['abc_transporter_ihc_score'].iloc[0] if 'abc_transporter_ihc_score' in tissue_data.columns else 1.0
        
        # Endothelial cells (node 6)
        vegf_level = blood_data['plasma_vegf_pg_ml'].iloc[0] if not blood_data.empty else 100
        node_features[6, 1] = vegf_level / 100.0
        
        # NK cells (node 7) - no direct data, infer from CD8
        node_features[7, 0] = cd8_density * 0.2
        
        # Dendritic cells (node 8) - infer from TLS presence
        tls_present = tissue_data['tertiary_lymphoid_structures'].iloc[0] if 'tertiary_lymphoid_structures' in tissue_data.columns else 0
        node_features[8, 0] = 100 if tls_present else 20
        
        # B cells (node 9) - infer from TLS
        node_features[9, 0] = 150 if tls_present else 30
        
        # Tregs (node 10) - correlate with MDSCs
        node_features[10, 0] = tissue_data['tissue_mdsc_density_per_mm2'].iloc[0] / 10.0 if 'tissue_mdsc_density_per_mm2' in tissue_data.columns else 10
        
        # Neutrophils (node 11)
        node_features[11, 0] = blood_data['circulating_mdsc_per_ml'].iloc[0] / 500.0 if not blood_data.empty else 10
        
        # Normalize features
        node_features = torch.nn.functional.normalize(node_features, p=2, dim=0)
        
        # Graph label: resistance mechanism (one-hot encoded)
        resistance_map = {
            'No Resistance': 0,
            'C797S': 1,
            'MET_amp': 2,
            'Loss_T790M':
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv
from torch_geometric.utils import from_networkx
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import json

class TMEGraphClassifier(nn.Module):
    """
    Graph Neural Network that classifies resistance mechanism from TME cell interactions
    Nodes = cell types, Edges = biological interactions
    """
    
    RESISTANCE_CLASSES = [
        'No Resistance',
        'C797S', 
        'MET_amp',
        'Loss_T790M',
        'Other'
    ]
    
    # Node types in TME graph
    NODE_TYPES = {
        0: 'Tumor_Cell',
        1: 'CD8_TIL',
        2: 'M2_TAM',
        3: 'MDSC',
        4: 'CAF',
        5: 'Endothelial_Cell'
    }
    
    # Interaction edges (biologically validated)
    EDGE_DEFINITIONS = [
        (0, 2, 'tumor_recruits_TAM'),      # Tumor -> M2 macrophage
        (0, 3, 'tumor_recruits_MDSC'),     # Tumor -> MDSC
        (2, 4, 'TAM_activates_CAF'),       # M2 TAM -> CAF
        (3, 1, 'MDSC_suppresses_TIL'),     # MDSC -> CD8 TIL
        (4, 0, 'CAF_supports_tumor'),      # CAF -> Tumor (HGF, TGF-β)
        (1, 0, 'TIL_kills_tumor'),         # CD8 TIL -> Tumor
        (5, 0, 'vascular_supplies_tumor'), # Endothelial -> Tumor
        (4, 5, 'CAF remodels_vasculature') # CAF -> Endothelial
    ]
    
    def __init__(self, node_feature_dim: int = 5, hidden_dim: int = 128):
        super().__init__()
        
        self.node_feature_dim = node_feature_dim
        
        # Graph convolution layers
        self.conv1 = GATConv(node_feature_dim, hidden_dim // 4, heads=4, dropout=0.1)
        self.conv2 = GATConv(hidden_dim, hidden_dim // 2, heads=3, dropout=0.1)
        self.conv3 = GCNConv(hidden_dim * 3 // 2, hidden_dim)
        
        # Global pooling
        self.pool = global_mean_pool
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, len(self.RESISTANCE_CLASSES))
        )
        
        # Uncertainty estimation module
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
    def build_tme_graph(self, patient_data: Dict) -> Data:
        """
        Convert patient biomarkers to TME graph
        """
        # Node features: [density, activation_score, cytokine_level, proximity, phenotype]
        node_features = []
        
        # Node 0: Tumor Cell
        node_features.append([
            patient_data.get('tumor_burden', 1e6) / 1e8,  # Normalized
            patient_data.get('proliferation_rate', 0.05),
            patient_data.get('hgf_level', 1.0),
            0.0,  # Self proximity
            1.0 if patient_data.get('resistance_type') == 'MET_amp' else 0.0
        ])
        
        # Node 1: CD8 TIL
        node_features.append([
            patient_data.get('cd8_density', 100) / 500,
            patient_data.get('cd8_activation', 0.5),
            0.0,  # No cytokine production in this model
            patient_data.get('cd8_tumor_distance', 100) / 200,
            0.0
        ])
        
        # Node 2: M2 TAM
        node_features.append([
            patient_data.get('m2_tam_density', 50) / 200,
            patient_data.get('m2_activation', 0.6),
            patient_data.get('hgf_level', 1.0) * 2.0,  # M2 produces HGF
            patient_data.get('m2_tumor_proximity', 50) / 100,
            0.0
        ])
        
        # Node 3: MDSC
        node_features.append([
            patient_data.get('mdsc_density', 30) / 150,
            patient_data.get('mdsc_suppression', 0.8),
            patient_data.get('il10_level', 1.0),
            patient_data.get('mdsc_tumor_proximity', 80) / 100,
            0.0
        ])
        
        # Node 4: CAF
        node_features.append([
            patient_data.get('caf_density', 80) / 250,
            patient_data.get('caf_activation', 0.7),
            patient_data.get('tgf_beta', 1.0) * 3.0,  # CAF produces TGF-β
            patient_data.get('caf_tumor_proximity', 30) / 100,
            0.0
        ])
        
        # Node 5: Endothelial
        node_features.append([
            patient_data.get('vessel_density', 150) / 500,
            patient_data.get('vascular_permeability', 0.4),
            patient_data.get('vegf_level', 1.0),
            50.0 / 100,
            0.0
        ])
        
        x = torch.tensor(node_features, dtype=torch.float32)
        
        # Edge indices
        edge_index = torch.tensor([
            [src for src, _, _ in self.EDGE_DEFINITIONS],
            [dst for _, dst, _ in self.EDGE_DEFINITIONS]
        ], dtype=torch.long)
        
        # Edge attributes (interaction strengths)
        edge_attr = torch.tensor([
            patient_data.get(f'{src}_{dst}_strength', 0.5) 
            for src, dst, _ in self.EDGE_DEFINITIONS
        ], dtype=torch.float32).reshape(-1, 1)
        
        # Labels
        resistance_idx = self.RESISTANCE_CLASSES.index(
            patient_data.get('resistance_mechanism', 'No Resistance')
        )
        y = torch.tensor(resistance_idx, dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: graph -> resistance class + uncertainty
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Graph convolutions
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        
        x = F.dropout(x, p=0.15, training=self.training)
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv3(x, edge_index)
        
        # Global pooling
        graph_representation = self.pool(x, batch)
        
        # Classification
        logits = self.classifier(graph_representation)
        
        # Uncertainty (epistemic + aleatoric)
        uncertainty = self.uncertainty_head(graph_representation)
        
        return logits, uncertainty
    
    def predict_from_patient_data(self, patient_data: Dict) -> Dict[str, float]:
        """
        Convenience method for direct patient data prediction
        """
        self.eval()
        with torch.no_grad():
            graph = self.build_tme_graph(patient_data)
            batch = torch.zeros(graph.x.size(0), dtype=torch.long)  # Single graph
            graph.batch = batch
            
            logits, uncertainty = self.forward(graph)
            probs = F.softmax(logits, dim=1).squeeze(0)  # Apply softmax and remove batch dim
            
            # Get top prediction
            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item()
            
            return {
                'predicted_mechanism': self.RESISTANCE_CLASSES[pred_idx],
                'confidence': confidence,
                'uncertainty': uncertainty.item(),
                'all_probabilities': {
                    cls: prob.item() 
                    for cls, prob in zip(self.RESISTANCE_CLASSES, probs)
                }
            }

class SyntheticTMEGrapher:
    """
    Generates synthetic TME graphs from your existing CSV data for GNN training
    """
    
    def __init__(self, ctdna_df: pd.DataFrame, tme_df: pd.DataFrame, tissue_df: pd.DataFrame):
        self.ctdna = ctdna_df
        self.tme = tme_df
        self.tissue = tissue_df
        
    def create_training_graphs(self) -> List[Data]:
        """
        Convert synthetic data to PyG graphs with augmented data and enhanced features
        """
        graphs = []
        
        for pid in self.tme['patient_id'].unique():
            # Get patient data
            patient_tme = self.tme[self.tme['patient_id'] == pid].iloc[0]
            patient_ctdna = self.ctdna[self.ctdna['patient_id'] == pid].iloc[0]
            
            # Base patient_data
            base_patient_data = self._create_patient_data(patient_tme, patient_ctdna)
            
            # Create original graph
            classifier = TMEGraphClassifier()
            graph = classifier.build_tme_graph(base_patient_data)
            graph.patient_id = pid
            graphs.append(graph)
            
            # Augment data: create 3 variations per patient for better training
            for aug_idx in range(3):
                augmented_data = self._augment_patient_data(base_patient_data, patient_ctdna['resistance_mechanism'])
                graph_aug = classifier.build_tme_graph(augmented_data)
                graph_aug.patient_id = f"{pid}_aug{aug_idx}"
                graphs.append(graph_aug)
        
        return graphs
    
    def _create_patient_data(self, patient_tme: pd.Series, patient_ctdna: pd.Series) -> Dict:
        """
        Create comprehensive patient_data dict with enhanced features
        """
        resistance = patient_ctdna['resistance_mechanism']
        
        # Base features
        patient_data = {
            # Tumor features
            'tumor_burden': 1e6 * (1 + patient_ctdna['ctdna_vaf_percent']),
            'proliferation_rate': 0.05,
            'resistance_mechanism': resistance,
            'baseline_vaf': patient_ctdna['ctdna_vaf_percent'],
            'resistance_type': resistance,  # For phenotype
            
            # Enhanced immune features
            'cd8_density': patient_tme.get('circulating_mdsc_per_ml', 100) * 0.5,
            'cd8_activation': 0.5,
            'cd8_tumor_distance': patient_tme.get('serum_crp_mg_l', 50) / 2,
            'cd4_density': patient_tme.get('circulating_mdsc_per_ml', 80) * 0.3,
            'b_cell_density': patient_tme.get('plasma_il10_pg_ml', 5) * 10,
            'nk_cell_density': patient_tme.get('serum_ldh_u_l', 200) / 500,
            
            # Enhanced myeloid features
            'm2_tam_density': patient_tme.get('circulating_mdsc_per_ml', 50) * 0.8,
            'm2_activation': 0.6,
            'm1_macrophage_density': patient_tme.get('circulating_mdsc_per_ml', 30) * 0.4,
            'mdsc_density': patient_tme['circulating_mdsc_per_ml'],
            'mdsc_suppression': min(patient_tme['plasma_il10_pg_ml'] / 10, 1.0),
            'mdsc_tumor_proximity': 80,
            'neutrophil_density': patient_tme.get('serum_ldh_u_l', 150) / 300,
            
            # Enhanced stromal features
            'caf_density': patient_tme.get('serum_tgfb_ng_ml', 10) * 5,
            'caf_activation': min(patient_tme['serum_tgfb_ng_ml'] / 20, 1.0),
            'caf_tumor_proximity': 30,
            'fibroblast_density': patient_tme.get('serum_tgfb_ng_ml', 15) * 3,
            
            # Enhanced vascular features
            'vessel_density': 150,
            'vascular_permeability': 0.4,
            'vegf_level': patient_tme['plasma_vegf_pg_ml'] / 100,
            'angiopoietin_level': patient_tme.get('plasma_vegf_pg_ml', 50) / 200,
            
            # Enhanced cytokines
            'hgf_level': patient_tme['serum_hgf_pg_ml'],
            'il10_level': patient_tme['plasma_il10_pg_ml'] / 10,
            'il6_level': patient_tme['plasma_il6_pg_ml'] / 10,
            'tgf_beta': patient_tme['serum_tgfb_ng_ml'],
            'ifn_gamma': patient_tme.get('plasma_il10_pg_ml', 2) * 0.5,  # Approximate
            'il1_beta': patient_tme.get('serum_crp_mg_l', 3) * 2,
            
            # Mechanism-specific interaction strengths
            '0_2_strength': self._get_interaction_strength(resistance, '0_2'),
            '0_3_strength': self._get_interaction_strength(resistance, '0_3'),
            '2_4_strength': self._get_interaction_strength(resistance, '2_4'),
            '3_1_strength': self._get_interaction_strength(resistance, '3_1'),
            '4_0_strength': self._get_interaction_strength(resistance, '4_0'),
            '1_0_strength': self._get_interaction_strength(resistance, '1_0'),
            '5_0_strength': self._get_interaction_strength(resistance, '5_0'),
            '4_5_strength': self._get_interaction_strength(resistance, '4_5')
        }
        
        # Mechanism-specific adjustments
        if resistance == 'Loss_T790M':
            patient_data['mdsc_density'] *= 1.5  # Boost MDSC for T790M
            patient_data['il10_level'] *= 1.2
            patient_data['mdsc_suppression'] = min(patient_data['mdsc_suppression'] * 1.3, 1.0)
        elif resistance == 'MET_amp':
            patient_data['caf_density'] *= 1.4
            patient_data['hgf_level'] *= 1.5
            patient_data['tgf_beta'] *= 1.2
        elif resistance == 'C797S':
            patient_data['m2_tam_density'] *= 1.3
            patient_data['il6_level'] *= 1.4
        # No Resistance and Other remain baseline
        
        return patient_data
    
    def _get_interaction_strength(self, resistance_mechanism: str, interaction_type: str) -> float:
        """
        Get mechanism-specific interaction strengths based on biological knowledge
        """
        base_strengths = {
            'No Resistance': {
                '0_2': 0.3, '0_3': 0.2, '2_4': 0.4, '3_1': 0.3,
                '4_0': 0.5, '1_0': 0.8, '5_0': 0.4, '4_5': 0.3
            },
            'C797S': {
                '0_2': 0.6, '0_3': 0.7, '2_4': 0.5, '3_1': 0.8,
                '4_0': 0.6, '1_0': 0.4, '5_0': 0.5, '4_5': 0.4
            },
            'MET_amp': {
                '0_2': 0.8, '0_3': 0.5, '2_4': 0.7, '3_1': 0.4,
                '4_0': 0.9, '1_0': 0.3, '5_0': 0.6, '4_5': 0.5
            },
            'Loss_T790M': {
                '0_2': 0.5, '0_3': 0.9, '2_4': 0.6, '3_1': 0.9,
                '4_0': 0.7, '1_0': 0.2, '5_0': 0.3, '4_5': 0.4
            },
            'Other': {
                '0_2': 0.4, '0_3': 0.4, '2_4': 0.4, '3_1': 0.4,
                '4_0': 0.4, '1_0': 0.4, '5_0': 0.4, '4_5': 0.4
            }
        }
        
        return base_strengths.get(resistance_mechanism, base_strengths['Other']).get(interaction_type, 0.5)
    
    def _augment_patient_data(self, base_data: Dict, resistance: str) -> Dict:
        """
        Create augmented version with slight perturbations for data augmentation
        """
        import numpy as np
        
        augmented = base_data.copy()
        
        # Add noise to continuous features
        noise_features = [
            'tumor_burden', 'cd8_density', 'm2_tam_density', 'mdsc_density', 'caf_density',
            'vessel_density', 'hgf_level', 'il10_level', 'il6_level', 'tgf_beta', 'vegf_level'
        ]
        
        for feature in noise_features:
            if feature in augmented:
                noise = np.random.normal(0, 0.1)  # 10% noise
                augmented[feature] *= (1 + noise)
                augmented[feature] = max(0, augmented[feature])  # Ensure non-negative
        
        # Slightly vary interaction strengths
        strength_keys = [k for k in augmented.keys() if 'strength' in k]
        for key in strength_keys:
            noise = np.random.normal(0, 0.05)  # Small variation
            augmented[key] = np.clip(augmented[key] + noise, 0.1, 1.0)
        
        return augmented

def train_gnn_classifier(graphs: List[Data], epochs: int = 100) -> Dict[str, float]:
    """
    Train GNN on synthetic TME graphs
    """
    # Split data
    train_size = int(0.8 * len(graphs))
    train_graphs = graphs[:train_size]
    val_graphs = graphs[train_size:]
    
    # Create DataLoader with batching
    train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=16, shuffle=False)
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TMEGraphClassifier().to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            logits, _ = model(batch)
            loss = criterion(logits, batch.y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
        
        train_losses.append(epoch_loss / len(train_loader))
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits, _ = model(batch)
                preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
                correct += (preds == batch.y).sum().item()
                total += batch.y.size(0)
        
        val_acc = correct / total
        val_accuracies.append(val_acc)
        scheduler.step(epoch_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {epoch_loss:.4f}, Val Acc = {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_accuracy': val_acc,
                'train_losses': train_losses,
                'val_accuracies': val_accuracies
            }, 'src/ml/checkpoints/tme_gnn_classifier.pth')
    
    return {
        'best_val_accuracy': best_val_acc,
        'final_train_loss': train_losses[-1],
        'train_losses': train_losses,
        'val_accuracies': val_accuracies
    }

# Example usage if run directly
if __name__ == "__main__":
    # Load synthetic data
    ctdna_df = pd.read_csv('src/data/synthetic_flaura2_ctdna.csv')
    tme_df = pd.read_csv('src/data/synthetic_tme_blood_factors.csv')
    tissue_df = pd.read_csv('src/data/synthetic_tissue_biopsy_data.csv')
    
    # Create graphs
    grapher = SyntheticTMEGrapher(ctdna_df, tme_df, tissue_df)
    graphs = grapher.create_training_graphs()
    
    print(f"Created {len(graphs)} TME graphs")
    print(f"Graph sample: {graphs[0]}")
    
    # Train GNN
    metrics = train_gnn_classifier(graphs, epochs=50)
    print(f"Training complete. Best validation accuracy: {metrics['best_val_accuracy']:.2%}")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import json

class PatientParameterNN(nn.Module):
    """
    Neural network to infer ODE parameters from patient biomarkers
    Maps 8-dimensional biomarker vector to 5 ODE parameters
    """
    
    # Physiological parameter ranges from literature_params.py
    PARAM_RANGES = {
        'r_R': (0.01, 0.1),      # Resistant growth rate
        'mu': (0.001, 0.5),      # Phenotypic plasticity rate
        'ABC': (0.5, 3.0),       # ABC transporter expression
        'sigma2': (0.1, 2.0),    # Epigenetic instability
        'K': (1e8, 1e10)         # Tumor carrying capacity
    }
    
    def __init__(self, input_dim: int = 8, hidden_dim: int = 64, dropout_rate: float = 0.2):
        super().__init__()
        
        # SIMPLIFIED ENCODER: Two layers instead of three
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # UNIFIED HEADS: Single multi-output layer instead of separate heads
        self.param_heads = nn.Linear(hidden_dim, len(self.PARAM_RANGES))
        
        # Initialize weights
        self._initialize_weights()
        
        # Store scalers for inference
        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        self.is_fitted = False
    
    def _initialize_weights(self):
        """Initialize network weights for stable training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)  # Lower gain for sigmoid
                nn.init.constant_(module.bias, 0.01)
    
    def forward(self, biomarkers: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass: biomarkers → ODE parameters
        
        Args:
            biomarkers: Tensor of shape [batch_size, input_dim]
            
        Returns:
            Dict of parameter tensors, each shape [batch_size, 1]
        """
        # Encode features
        features = self.encoder(biomarkers)
        
        # Get raw parameter predictions from unified heads
        raw_params = self.param_heads(features)
        
        # Apply sigmoid and scale to parameter ranges
        params = {}
        param_names = list(self.PARAM_RANGES.keys())
        
        for i, name in enumerate(param_names):
            # Sigmoid activation for bounded outputs
            sigmoid_out = torch.sigmoid(raw_params[:, i])
            
            # Scale to parameter range
            min_val, max_val = self.PARAM_RANGES[name]
            scaled = sigmoid_out * (max_val - min_val) + min_val
            
            # Log transform for carrying capacity (K) to handle large range
            if name == 'K':
                scaled = 10 ** (scaled / 1e9)  # Proper log scaling
            
            params[name] = scaled.unsqueeze(1)  # Add dimension for consistency
        
        return params
    
    def predict_from_pandas(self, biomarkers_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict parameters from pandas DataFrame without gradient tracking
        
        Args:
            biomarkers_df: DataFrame with feature columns
            
        Returns:
            DataFrame with predicted parameters (in original scale)
        """
        self.eval()
        with torch.no_grad():
            X = torch.tensor(biomarkers_df.values, dtype=torch.float32)
            
            # Scale if fitted
            if self.is_fitted:
                X_np = X.numpy()
                X_scaled = self.feature_scaler.transform(X_np)
                X = torch.tensor(X_scaled, dtype=torch.float32)
            
            # Get predictions in original physiological scale
            params_pred = self.forward(X)
            
            # Convert to DataFrame
            param_names = list(self.PARAM_RANGES.keys())
            pred_dict = {name: params_pred[name].numpy().flatten() for name in param_names}
            result = pd.DataFrame(pred_dict, index=biomarkers_df.index)
        
        return result

class PatientParameterDataset(Dataset):
    """Dataset wrapper for training data"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = features
        self.targets = targets
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.features[idx], dtype=torch.float32),
            'targets': torch.tensor(self.targets[idx], dtype=torch.float32)
        }

def prepare_training_data(model: PatientParameterNN, ctdna_df: pd.DataFrame, tme_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare biomarker features and ODE parameter targets from synthetic data
    with ROBUST SCALING that handles mixed parameter scales properly.
    """
    # Merge datasets on patient and week
    merged = ctdna_df.merge(tme_df, on=['patient_id', 'week'], how='inner')
    merged['resistance_mechanism'] = merged['resistance_mechanism_x']
    merged.drop(columns=['resistance_mechanism_x', 'resistance_mechanism_y'], inplace=True)
    if merged.empty:
        raise ValueError("Merge produced empty dataset - check column names")
    
    # Get baseline (week 0) data
    baseline = merged[merged['week'] == 0]
    
    if baseline.empty:
        raise ValueError("No baseline data found for week=0")
    
    # Feature preparation
    feature_columns = [
        'ctdna_vaf_percent',
        'serum_hgf_pg_ml',
        'plasma_il6_pg_ml',
        'circulating_mdsc_per_ml',
        'serum_tgfb_ng_ml',
        'ctc_count_per_ml',
        'serum_crp_mg_l',
        'serum_ldh_u_l'
    ]
    
    # Check for missing columns
    missing_features = [col for col in feature_columns if col not in baseline.columns]
    if missing_features:
        warnings.warn(f"Missing feature columns: {missing_features}. Filling with zeros.")
        for col in missing_features:
            baseline[col] = 0.0
    
    features = baseline[feature_columns].fillna(0).values
    
    # Parameter target preparation
    param_targets = []
    
    for pid in baseline['patient_id'].unique():
        patient_ctdna = merged[merged['patient_id'] == pid]
        
        if len(patient_ctdna) < 2:
            # Default parameters for insufficient data
            param_targets.append([0.05, 0.05, 1.0, 0.5, 1e9])
            continue
        
        # Infer r_R from time to progression (more realistic)
        ttp_months = patient_ctdna['ttp_months'].iloc[0] if 'ttp_months' in patient_ctdna.columns else 24.0
        # r_R should be higher for shorter TTP (faster progression)
        r_R = max(0.01, min(0.1, 0.12 - 0.002 * ttp_months))  # Linear relationship
        
        # Infer mu from ctDNA clearance (improved logic)
        baseline_vaf = patient_ctdna['ctdna_vaf_percent'].iloc[0]
        week6_vaf = patient_ctdna[patient_ctdna['week'] == 6]['ctdna_vaf_percent'].iloc[0] if len(patient_ctdna[patient_ctdna['week'] == 6]) > 0 else baseline_vaf
        clearance_rate = max(0.0, min(1.0, (baseline_vaf - week6_vaf) / max(baseline_vaf, 0.001)))
        # Higher clearance = higher drug sensitivity (mu)
        mu = max(0.001, min(0.5, 0.1 + 0.4 * clearance_rate))
        
        # Infer ABC from resistance mechanism (more systematic)
        if 'resistance_mechanism' in patient_ctdna.columns:
            resistance = patient_ctdna['resistance_mechanism'].iloc[0]
            abc_dict = {'MET_amp': 2.8, 'C797S': 2.2, 'T790M': 1.8, 'exon19del': 1.2, 'L858R': 1.1}
            ABC = abc_dict.get(resistance, 1.0 + np.random.uniform(0, 0.5))
        else:
            ABC = 1.0
            warnings.warn(f"Missing resistance_mechanism for {pid}, using ABC=1.0")
        
        # Carrying capacity K: Higher for more aggressive tumors
        K_base = 5e8 + np.random.uniform(0, 5e8)  # Base range
        K = K_base * (1 + 0.5 * (r_R - 0.01) / 0.09)  # Scale with growth rate
        K = max(1e8, min(1e10, K))
        
        # Plasticity sigma2: Higher for more variable ctDNA
        vaf_std = patient_ctdna['ctdna_vaf_percent'].std() if len(patient_ctdna) > 1 else 0.1
        sigma2 = max(0.1, min(2.0, 0.2 + 2.0 * vaf_std / 10.0))
        
        param_targets.append([r_R, mu, ABC, sigma2, K])
    
    param_targets = np.array(param_targets)
    
    # FIXED SCALING: Only scale features, keep targets in original physiological ranges
    # This avoids double-scaling confusion and maintains interpretability
    feature_scaler = StandardScaler()
    
    features_scaled = feature_scaler.fit_transform(features)
    # Keep targets in original scale - network will learn to predict in physiological ranges
    
    # Store scalers in model for inference
    model.feature_scaler = feature_scaler
    model.target_scaler = None  # Not using target scaling anymore
    model.is_fitted = True
    
    return (torch.tensor(features_scaled, dtype=torch.float32), 
            torch.tensor(param_targets, dtype=torch.float32))

def train_model(model: PatientParameterNN, 
                X_train: torch.Tensor, 
                y_train: torch.Tensor,
                X_val: torch.Tensor = None,
                y_val: torch.Tensor = None,
                epochs: int = 100,
                lr: float = 1e-3,
                batch_size: int = 16,
                patience: int = 15,
                checkpoint_dir: str = 'src/ml/checkpoints') -> Dict[str, float]:
    """
    Train PatientParameterNN with MSE loss - FIXED for numerical stability
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()
    
    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Create data loaders
    train_dataset = PatientParameterDataset(X_train.numpy(), y_train.numpy())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    if X_val is not None and y_val is not None:
        val_dataset = PatientParameterDataset(X_val.numpy(), y_val.numpy())
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None
    
    # Optimizer - use lower LR and weight decay for stability
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    # FIXED: Remove deprecated 'verbose' parameter for PyTorch compatibility
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Loss tracking
    losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch in train_loader:
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)
            
            # Forward pass
            pred_params = model(features)
            param_names = list(model.PARAM_RANGES.keys())
            pred_tensor = torch.cat([pred_params[name] for name in param_names], dim=1)
            
            # Compute MSE loss directly on physiological scale
            # Use log-space MSE for K to handle large range
            loss_r_mu_abc_sigma2 = torch.mean((pred_tensor[:, :4] - targets[:, :4])**2)
            loss_K = torch.mean((torch.log10(pred_tensor[:, 4]) - torch.log10(targets[:, 4]))**2)
            loss = loss_r_mu_abc_sigma2 + loss_K
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        losses.append(avg_train_loss)
        
        # Validation - FIXED: use .item() to get scalar
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    features = batch['features'].to(device)
                    targets = batch['targets'].to(device)
                    
                    pred_params = model(features)
                    pred_tensor = torch.cat([pred_params[name] for name in param_names], dim=1)
                    # Use same loss computation as training
                    loss_r_mu_abc_sigma2 = torch.mean((pred_tensor[:, :4] - targets[:, :4])**2)
                    loss_K = torch.mean((torch.log10(pred_tensor[:, 4]) - torch.log10(targets[:, 4]))**2)
                    loss = loss_r_mu_abc_sigma2 + loss_K
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            # FIX: Pass scalar to scheduler
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # Save checkpoint with scalers
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'losses': losses,
                    'feature_scaler': model.feature_scaler,
                    'target_scaler': None,  # Not using target scaling
                    'is_fitted': True
                }
                torch.save(checkpoint, f"{checkpoint_dir}/patient_parameter_nn.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {avg_train_loss:.6f}")
    
    return {'final_loss': losses[-1]}

# ============================================================================
# MAIN EXECUTION BLOCK
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("PatientParameterNN Training Pipeline")
    print("=" * 60)
    
    # Check for data files
    ctdna_file = Path('src/data/synthetic_flaura2_ctdna.csv')
    tme_file = Path('src/data/synthetic_tme_blood_factors.csv')
    
    if not ctdna_file.exists() or not tme_file.exists():
        print("❌ Synthetic data files not found. Please generate them first.")
        exit()
    
    # Load data
    print(f"\n[1/4] Loading synthetic data...")
    ctdna_df = pd.read_csv(ctdna_file)
    tme_df = pd.read_csv(tme_file)
    print(f"   - ctDNA samples: {len(ctdna_df)}")
    print(f"   - Patients: {ctdna_df['patient_id'].nunique()}")
    
    # Preprocess
    print(f"\n[2/4] Preprocessing training data...")
    model = PatientParameterNN()
    X, y = prepare_training_data(model, ctdna_df, tme_df)
    print(f"   - Features shape: {X.shape}")
    print(f"   - Targets shape: {y.shape}")
    
    # Split train/val
    print(f"\n[3/4] Splitting train/validation...")
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    print(f"   - Train: {len(X_train)} samples")
    print(f"   - Val: {len(X_val)} samples")
    
    # Initialize model
    print(f"\n[4/4] Initializing and training model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PatientParameterNN(hidden_dim=128, dropout_rate=0.3)
    model = model.to(device)
    print(f"   - Device: {device}")
    print(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    metrics = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=150,
        lr=1e-3,
        batch_size=16,
        patience=20
    )
    
    print(f"\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final Train Loss: {metrics['final_loss']:.6f}")
    print(f"Model saved with robust scaling for inference.")
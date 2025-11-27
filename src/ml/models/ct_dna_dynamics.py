from pathlib import Path
import torch
import torch.nn as nn
from torchdiffeq import odeint
import numpy as np
from typing import Dict, Any, List, Optional, Union
import warnings
import pandas as pd

class ctDNANeuralODE(nn.Module):
    """
    Neural ODE for ctDNA dynamics: models generation and clearance
    Couples with tumor population ODE from original model
    
    Args:
        hidden_dim: Dimension of hidden layers in production network
        device: torch device ('cpu' or 'cuda')
    """
    
    def __init__(self, hidden_dim: int = 32, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or torch.device('cpu')
        
        # Neural network for ctDNA production rate
        # Inputs: [ctDNA, tumor_burden, tumor_rate, clone_fraction, drug_concentration] (5 inputs)
        self.production_net = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.ReLU()  # Production is non-negative
        ).to(self.device)
        
        # Clearance rate (learnable parameter)
        self.clearance_rate = nn.Parameter(torch.tensor(0.1, device=self.device))  # ~10% per day
        
        # NEW: Add initialization bounds
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights for stable training"""
        for module in self.production_net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                nn.init.constant_(module.bias, 0.1)
    
    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of ctDNA dynamics
        
        Args:
            t: Time point (scalar tensor)
            state: [ctDNA_concentration, tumor_burden, tumor_burden_rate, clone_fraction, drug_concentration]
        
        Returns:
            d(state)/dt (tensor of shape [5])
        """
        if state.ndim != 1 or state.size(0) != 5:
            raise ValueError(f"Expected state shape [5], got {state.shape}")
        
        ctDNA, tumor_burden, tumor_rate, clone_frac, drug = state
        
        # Use neural network to predict ctDNA production rate from full state
        state_tensor = state.reshape(1, -1).to(self.device)
        production = self.production_net(state_tensor).squeeze()
        
        # Clearance proportional to current ctDNA
        clearance = self.clearance_rate * ctDNA
        
        d_ctdna_dt = production - clearance
        
        # Return derivatives for all 5 state variables
        # Tumor state variables are held constant (derivatives = 0)
        # Only ctDNA changes
        derivatives = torch.zeros_like(state, device=self.device)
        derivatives[0] = d_ctdna_dt  # d(ctDNA)/dt
        
        return derivatives
    
    def simulate(self, 
                 initial_ctdna: float, 
                 tumor_trajectory: torch.Tensor, 
                 time_points: torch.Tensor,
                 drug_concentration: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Simulate ctDNA over time given tumor trajectory
        
        Args:
            initial_ctdna: Baseline ctDNA concentration (float)
            tumor_trajectory: Tensor of shape [T, 3] with [burden, rate, clone_frac]
            time_points: Tensor of time points [T]
            drug_concentration: Optional drug concentration over time [T]
        
        Returns:
            ctDNA trajectory: Tensor [T]
        """
        # Validate inputs
        if tumor_trajectory.ndim != 2 or tumor_trajectory.size(1) != 3:
            raise ValueError(f"Expected tumor_trajectory shape [T, 3], got {tumor_trajectory.shape}")
        
        if time_points.size(0) != tumor_trajectory.size(0):
            raise ValueError("time_points and tumor_trajectory must have same length")
        
        # Default drug concentration if not provided
        if drug_concentration is None:
            drug_concentration = torch.ones_like(time_points) * 1.0  # 1.0 μM baseline
        
        # Initial state: [ctDNA, tumor_burden, tumor_rate, clone_frac, drug]
        y0 = torch.tensor([
            initial_ctdna, 
            tumor_trajectory[0, 0],  # Initial burden
            tumor_trajectory[0, 1],  # Initial rate
            tumor_trajectory[0, 2],  # Initial clone fraction
            drug_concentration[0]    # Initial drug
        ], dtype=torch.float32, device=self.device)
        
        # Solve ODE
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # Suppress torchdiffeq warnings
            ctDNA_traj = odeint(self, y0, time_points.to(self.device), method='dopri5')
        
        return ctDNA_traj[:, 0]  # Return only ctDNA concentration

def integrate_with_original_ode(original_ode_func, ctdna_ode, initial_state, time_points):
    """
    Couple the original tumor ODE with ctDNA ODE
    
    Args:
        original_ode_func: Function(t, y) returning tumor dynamics (numpy array)
        ctdna_ode: Instance of ctDNANeuralODE
        initial_state: Initial state [S, R, D, ABC, sigma2] (list or numpy array)
        time_points: Time points for integration
        
    Returns:
        Combined trajectory: [S, R, D, ABC, sigma2, ctDNA] (numpy array)
    """
    class CoupledODE(nn.Module):
        def __init__(self, original_ode, ctdna_ode):
            super().__init__()
            self.original_ode = original_ode
            self.ctdna_ode = ctdna_ode
            
        def forward(self, t, y):
            # y = [S, R, D, ABC, sigma2, ctDNA]
            # Extract tumor and ctDNA states
            tumor_state = y[:5]
            ctDNA = y[5]
            
            # Compute tumor dynamics (convert to numpy then back to torch)
            tumor_dynamics_np = self.original_ode(t.cpu(), tumor_state.cpu().numpy())
            tumor_dynamics = torch.from_numpy(tumor_dynamics_np).float().to(ctDNA.device)
            
            # Compute tumor burden and rate for ctDNA ODE
            tumor_burden = tumor_state[0] + tumor_state[1]  # S + R
            tumor_rate = tumor_dynamics[0] + tumor_dynamics[1]
            clone_fraction = torch.clamp(tumor_state[1] / (tumor_burden + 1e-10), 0, 1)
            
            # Compute ctDNA dynamics
            drug_concentration = tumor_state[2]
            ctDNA_state = torch.cat([
                ctDNA.reshape(1),
                tumor_burden.reshape(1),
                tumor_rate.reshape(1),
                clone_fraction.reshape(1),
                drug_concentration.reshape(1)
            ])
            
            d_ctdna_dt = self.ctdna_ode.forward(t, ctDNA_state)
            
            # Combine dynamics
            return torch.cat([tumor_dynamics, d_ctdna_dt.reshape(1)])
    
    # Initial state: [S, R, D, ABC, sigma2, ctDNA]
    if isinstance(initial_state, list):
        initial_state = np.array(initial_state)
    
    y0 = np.concatenate([initial_state, [0.001]])  # Add small initial ctDNA
    y0_torch = torch.from_numpy(y0).float().to(ctdna_ode.device)
    
    # Solve coupled system
    try:
        trajectory_torch = odeint(CoupledODE(original_ode_func, ctdna_ode), 
                                y0_torch, time_points.to(ctdna_ode.device), 
                                method='lsoda')
        return trajectory_torch.detach().cpu().numpy()
    except Exception as e:
        warnings.warn(f"Coupled ODE integration failed: {e}. Returning tumor-only trajectory.")
        # Fallback: return tumor trajectory with zero ctDNA
        tumor_traj = original_ode_func(time_points, y0_torch[:5])
        ctDNA_zeros = np.zeros_like(time_points)
        return np.concatenate([tumor_traj, ctDNA_zeros.reshape(-1, 1)], axis=1)

def simulate_with_ctdna(params, initial_state, treatment_protocol, weeks=52):
    """
    Wrapper that runs original ODE simulation and generates ctDNA predictions
    
    Args:
        params: Model parameters dictionary
        initial_state: Initial tumor state [S, R, D, ABC, sigma2]
        treatment_protocol: Drug schedule function
        weeks: Simulation duration
        
    Returns:
        Dictionary with days, tumor, ctdna_vaf, ctdna_concentration
    """
    from src.models.tumour_population import run_simulation  # Local import to avoid circular dependency
    
    # Time points (daily resolution)
    days = np.arange(0, weeks * 7 + 1, 1)
    
    # Run original tumor simulation
    tumor_traj = run_simulation(params, initial_state, treatment_protocol, days)
    
    # Initialize ctDNA model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ctdna_model = ctDNANeuralODE(device=device)
    ctdna_model.eval()
    
    # Prepare tumor trajectory tensor [T, 3]
    tumor_tensor = torch.tensor([
        [tumor_traj['S'][i] + tumor_traj['R'][i],  # Total burden
         0,  # Rate (simplified - could be improved with gradient)
         tumor_traj['R'][i] / max(tumor_traj['S'][i] + tumor_traj['R'][i], 1)  # Resistant fraction
        ] for i in range(len(days))
    ], dtype=torch.float32, device=device)
    
    time_tensor = torch.tensor(days, dtype=torch.float32, device=device)
    
    # Simulate ctDNA
    with torch.no_grad():
        baseline_ctdna = params.get('baseline_ctdna', 0.001)  # Default 0.1% VAF
        ctDNA_pred = ctdna_model.simulate(baseline_ctdna, tumor_tensor, time_tensor)
    
    # Convert to VAF (simplified: 1 ctDNA molecule per 1000 tumor cells ~ 0.1% VAF)
    # This is a physiologically reasonable approximation
    ctDNA_vaf = (ctDNA_pred.cpu().numpy() / (tumor_tensor[:, 0].cpu().numpy() / 1000)) * 100
    
    return {
        'days': days,
        'tumor': tumor_traj,
        'ctdna_vaf': ctDNA_vaf,
        'ctdna_concentration': ctDNA_pred.cpu().numpy()
    }

# ============================================================================
# DATA PREPROCESSING & TRAINING FUNCTIONS
# ============================================================================

def preprocess_training_data(ctdna_df: pd.DataFrame, tme_df: pd.DataFrame) -> List[Dict[str, torch.Tensor]]:
    """
    Convert synthetic ctDNA data into training pairs for Neural ODE
    
    Training data consists of:
    - Input: Tumor trajectory (burden, growth rate, clone fraction) + drug concentration over time
    - Target: ctDNA VAF measurements over time
    
    Args:
        ctdna_df: DataFrame with ctDNA time series
        tme_df: DataFrame with TME biomarkers
        
    Returns:
        List of dictionaries with tensors for each patient
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    training_pairs = []
    
    for patient_id in ctdna_df['patient_id'].unique():
        # Extract patient time series (ensure sorted by time)
        patient_ctdna = ctdna_df[ctdna_df['patient_id'] == patient_id].sort_values('week')
        patient_tme = tme_df[tme_df['patient_id'] == patient_id].sort_values('week')
        
        if len(patient_ctdna) < 3:  # Need at least 3 timepoints for gradient
            continue
        
        # Time points (convert weeks to days)
        time_points = torch.tensor(patient_ctdna['week'].values * 7, dtype=torch.float32, device=device)
        
        # Target: actual ctDNA VAF
        ctdna_targets = torch.tensor(patient_ctdna['ctdna_vaf_percent'].values, dtype=torch.float32, device=device)
        
        # Features: approximate tumor burden from ctDNA molecules
        ctdna_molecules = torch.tensor(patient_ctdna['ctdna_molecules_per_ml'].values, dtype=torch.float32, device=device)
        tumor_burden = ctdna_molecules * 1000  # Approximate scaling factor
        
        # Compute tumor growth rate (numerical gradient)
        tumor_rate = torch.zeros_like(tumor_burden)
        if len(tumor_burden) > 2:
            tumor_rate[1:-1] = (tumor_burden[2:] - tumor_burden[:-2]) / (time_points[2:] - time_points[:-2])
            tumor_rate[0] = tumor_rate[1]
            tumor_rate[-1] = tumor_rate[-2]
        
        # Resistant clone fraction (ctDNA VAF normalized)
        clone_fraction = torch.clamp(ctdna_targets / 100, min=0.0, max=1.0)
        
        # Create tumor trajectory tensor [T, 3]
        tumor_trajectory = torch.stack([
            tumor_burden,
            tumor_rate,
            clone_fraction
        ], dim=1)
        
        # Drug concentration (placeholder - in real data this would come from regimen)
        drug_concentration = torch.ones_like(time_points) * 5.0
        
        training_pairs.append({
            'time_points': time_points,
            'tumor_trajectory': tumor_trajectory,
            'drug_concentration': drug_concentration,
            'ctdna_targets': ctdna_targets,
            'patient_id': patient_id
        })
    
    return training_pairs

def train_model(model: ctDNANeuralODE,
                training_data: List[Dict[str, torch.Tensor]],
                learning_rate: float = 1e-4,
                epochs: int = 100,
                batch_size: int = 8,
                validation_split: float = 0.2,
                patience: int = 10,
                checkpoint_dir: str = 'src/ml/checkpoints') -> Dict[str, Any]:
    """
    Train ctDNANeuralODE on preprocessed data
    
    Args:
        model: Model to train
        training_data: List of training pairs from preprocess_training_data
        learning_rate: AdamW learning rate
        epochs: Max training epochs
        batch_size: Number of patients per batch
        validation_split: Fraction of patients for validation
        patience: Early stopping patience
        checkpoint_dir: Directory to save checkpoints
        
    Returns:
        Dict with training metrics
    """
    
    device = model.device
    model.train()
    
    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Split data
    n_patients = len(training_data)
    split_idx = int(n_patients * (1 - validation_split))
    train_data = training_data[:split_idx]
    val_data = training_data[split_idx:]
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Training tracking
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        
        # Shuffle training data
        np.random.shuffle(train_data)
        
        for i in range(0, len(train_data), batch_size):
            batch_patients = train_data[i:i+batch_size]
            
            optimizer.zero_grad()
            batch_loss = 0.0
            
            for patient in batch_patients:
                # Get current state for each time point
                batch_loss = 0.0
                for t_idx in range(len(patient['time_points'])):
                    # Current state: [ctDNA, tumor_burden, tumor_rate, clone_frac, drug]
                    current_state = torch.tensor([
                        patient['ctdna_targets'][t_idx-1] if t_idx > 0 else 0.1,  # Previous ctDNA or initial
                        patient['tumor_trajectory'][t_idx, 0],  # Current tumor burden
                        patient['tumor_trajectory'][t_idx, 1],  # Current tumor rate
                        patient['tumor_trajectory'][t_idx, 2],  # Current clone fraction
                        patient['drug_concentration'][t_idx]    # Current drug concentration
                    ], dtype=torch.float32, device=device)
                    
                    # Predict ctDNA production rate
                    pred_production = model.production_net(current_state.unsqueeze(0)).squeeze()
                    
                    # Target: rate of change in ctDNA (simplified as difference)
                    if t_idx > 0:
                        dt = patient['time_points'][t_idx] - patient['time_points'][t_idx-1]
                        actual_ctdna_change = patient['ctdna_targets'][t_idx] - patient['ctdna_targets'][t_idx-1]
                        target_rate = actual_ctdna_change / dt
                        
                        # Loss on predicted production rate
                        loss = criterion(pred_production, target_rate)
                        batch_loss += loss
                
                batch_loss = batch_loss / max(1, len(patient['time_points']) - 1)
            
            # Average loss over batch
            batch_loss = batch_loss / len(batch_patients)
            batch_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_train_loss += batch_loss.item()
        
        avg_train_loss = epoch_train_loss / (len(train_data) // batch_size + 1)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        
        with torch.no_grad():
            for patient in val_data:
                patient_loss = 0.0
                for t_idx in range(len(patient['time_points'])):
                    # Current state: [ctDNA, tumor_burden, tumor_rate, clone_frac, drug]
                    current_state = torch.tensor([
                        patient['ctdna_targets'][t_idx-1] if t_idx > 0 else 0.1,  # Previous ctDNA or initial
                        patient['tumor_trajectory'][t_idx, 0],  # Current tumor burden
                        patient['tumor_trajectory'][t_idx, 1],  # Current tumor rate
                        patient['tumor_trajectory'][t_idx, 2],  # Current clone fraction
                        patient['drug_concentration'][t_idx]    # Current drug concentration
                    ], dtype=torch.float32, device=device)
                    
                    # Predict ctDNA production rate
                    pred_production = model.production_net(current_state.unsqueeze(0)).squeeze()
                    
                    # Target: rate of change in ctDNA
                    if t_idx > 0:
                        dt = patient['time_points'][t_idx] - patient['time_points'][t_idx-1]
                        actual_ctdna_change = patient['ctdna_targets'][t_idx] - patient['ctdna_targets'][t_idx-1]
                        target_rate = actual_ctdna_change / dt
                        
                        # Loss on predicted production rate
                        loss = criterion(pred_production, target_rate)
                        patient_loss += loss.item()
                
                patient_loss = patient_loss / max(1, len(patient['time_points']) - 1)
                epoch_val_loss += patient_loss
        
        avg_val_loss = epoch_val_loss / len(val_data)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_epoch = epoch
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'hyperparameters': {
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'validation_split': validation_split
                }
            }, f"{checkpoint_dir}/ctdna_neuralode_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
                break
        
        # Print progress
        if epoch % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}, LR = {current_lr:.6f}")
    
    return {
        'final_train_loss': train_losses[-1],
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'epochs_trained': len(train_losses)
    }

# ============================================================================
# MAIN EXECUTION BLOCK (Replace Existing)
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("ctDNANeuralODE Training Pipeline")
    print("=" * 60)
    
    
    ctdna_file = "src/data/synthetic_flaura2_ctdna.csv"
    tme_file = "src/data/synthetic_tme_blood_factors.csv"
    
    # Load data
    print(f"\n[1/4] Loading synthetic data...")
    ctdna_df = pd.read_csv(ctdna_file)
    tme_df = pd.read_csv(tme_file)
    print(f"   - ctDNA samples: {len(ctdna_df)}")
    print(f"   - Patients: {ctdna_df['patient_id'].nunique()}")
    
    # Preprocess
    print(f"\n[2/4] Preprocessing training data...")
    training_pairs = preprocess_training_data(ctdna_df, tme_df)
    print(f"   - Training pairs: {len(training_pairs)}")
    
    # Initialize model
    print(f"\n[3/4] Initializing model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ctDNANeuralODE(hidden_dim=64, device=device)
    print(f"   - Device: {device}")
    print(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    print(f"\n[4/4] Starting training...")
    metrics = train_model(
        model=model,
        training_data=training_pairs,
        learning_rate=1e-4,
        epochs=100,
        batch_size=4,
        validation_split=0.2,
        patience=15
    )
    
    print(f"\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best Validation Loss: {metrics['best_val_loss']:.6f}")
    print(f"Best Epoch: {metrics['best_epoch']}")
    print(f"Final Train Loss: {metrics['final_train_loss']:.6f}")
    print(f"Model saved to: src/ml/checkpoints/ctdna_neuralode_best.pth")
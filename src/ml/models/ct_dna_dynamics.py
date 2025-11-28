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
    
    Literature basis:
    - Diehl et al. PNAS 2008: ctDNA half-life ~1.5 hours (clearance rate ~11 per day)
    - Bettegowda et al. Sci Transl Med 2014: ctDNA production proportional to tumor cell death
    - Wan et al. Nat Rev Clin Oncol 2017: ctDNA kinetics reflect tumor burden and treatment response
    - Chaudhuri et al. Nat Med 2017: ctDNA clearance follows exponential decay with t1/2 ~1-2 hours
    
    Mathematical model:
    d(ctDNA)/dt = k_production * cell_death_rate - k_clearance * ctDNA
    
    Where:
    - k_clearance = ln(2) / t_half ≈ 11.1 per day (for t_half = 1.5 hours)
    - k_production is learned from data (proportional to cell death)
    
    Args:
        hidden_dim: Dimension of hidden layers in production network
        device: torch device ('cpu' or 'cuda')
    """
    
    def __init__(self, hidden_dim: int = 32, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or torch.device('cpu')
        
        # Neural network for ctDNA production rate
        # Inputs: [log_burden/10, death_rate*10, clone_fraction, drug/10] (4 inputs)
        # Outputs: log10(ctDNA) - transform to actual ctDNA via 10^output
        self.production_net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
            # No activation - output is log10(ctDNA) which can be negative
        ).to(self.device)
        
        # Clearance rate parameter (log scale for positivity)
        # Literature: ctDNA half-life ~1.5 hours = 0.0625 days
        # k_clearance = ln(2) / t_half = ln(2) / 0.0625 ≈ 11.1 per day
        # Initialize to log(11.1) ≈ 2.4 (was -3.0 which gave 0.05 - too slow!)
        self.clearance_log = nn.Parameter(torch.tensor(2.4))  # exp(2.4) ≈ 11.0 per day
        
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
        Forward pass of ctDNA VAF dynamics
        
        Literature-based model:
        d(ctDNA)/dt = k_production * cell_death_rate - k_clearance * ctDNA
        
        Where:
        - k_clearance ≈ 11 per day (half-life ~1.5 hours, Diehl PNAS 2008)
        - Production is proportional to tumor cell death (baseline + drug-induced)
        
        Args:
            t: Time point (scalar tensor)
            state: [ctDNA_VAF, tumor_burden, tumor_death_rate, clone_fraction, drug_concentration]
                   Note: tumor_death_rate should include both baseline apoptosis and drug kill
        
        Returns:
            d(state)/dt (tensor of shape [5])
        """
        if state.ndim != 1 or state.size(0) != 5:
            raise ValueError(f"Expected state shape [5], got {state.shape}")
        
        vaf, tumor_burden, tumor_death_rate, clone_frac, drug = state
        
        # Use neural network to predict ctDNA production rate
        # Input: [tumor_burden, tumor_death_rate, clone_frac, drug]
        # Production should scale with cell death rate (baseline + drug-induced)
        # Normalize inputs for better training stability
        production_input = torch.stack([
            tumor_burden / 1e8,  # Normalize to typical tumor size
            torch.clamp(tumor_death_rate, 0, 1.0),  # Death rate (fraction per day)
            clone_frac,  # Already 0-1
            drug / 10.0  # Normalize drug concentration
        ]).to(self.device)
        production_rate = self.production_net(production_input.unsqueeze(0)).squeeze()
        
        # Clearance rate based on literature (Diehl PNAS 2008, Bettegowda Sci Transl Med 2014)
        # ctDNA half-life ~1.5 hours = 0.0625 days
        # k_clearance = ln(2) / t_half ≈ 11.1 per day
        # Allow slight variation around this value (±20%)
        clearance_rate = torch.exp(self.clearance_log)
        # Clamp to physiologically reasonable range: 8-16 per day
        clearance_rate = torch.clamp(clearance_rate, 8.0, 16.0)
        
        # ctDNA dynamics: d(ctDNA)/dt = production - clearance * ctDNA
        # This follows first-order kinetics as established in literature
        d_vaf_dt = production_rate - clearance_rate * vaf
        
        # Return derivatives for all 5 state variables
        # Tumor state variables are held constant (derivatives = 0)
        # Only VAF changes
        derivatives = torch.zeros_like(state, device=self.device)
        derivatives[0] = d_vaf_dt  # d(VAF)/dt
        
        return derivatives
    
    def simulate(self, 
                 initial_vaf: float, 
                 tumor_trajectory: torch.Tensor, 
                 time_points: torch.Tensor,
                 drug_concentration: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Simulate ctDNA VAF over time given tumor trajectory
        
        Args:
            initial_vaf: Baseline ctDNA VAF (%)
            tumor_trajectory: Tensor of shape [T, 3] or [T, 4]
                - [T, 3]: [burden, growth_rate, clone_frac] (legacy format)
                - [T, 4]: [burden, death_rate, clone_frac, drug] (new format with death rate)
            time_points: Tensor of time points [T]
            drug_concentration: Optional drug concentration over time [T] (ignored if in trajectory)
        
        Returns:
            ctDNA VAF trajectory: Tensor [T]
        """
        # Validate inputs
        if tumor_trajectory.ndim != 2:
            raise ValueError(f"Expected tumor_trajectory to be 2D, got {tumor_trajectory.ndim}D")
        
        n_features = tumor_trajectory.size(1)
        if n_features not in [3, 4]:
            raise ValueError(f"Expected tumor_trajectory shape [T, 3] or [T, 4], got {tumor_trajectory.shape}")
        
        if time_points.size(0) != tumor_trajectory.size(0):
            raise ValueError("time_points and tumor_trajectory must have same length")
        
        # Handle both legacy [T, 3] and new [T, 4] formats
        if n_features == 4:
            # New format: [burden, death_rate, clone_frac, drug]
            drug_concentration = tumor_trajectory[:, 3]
            tumor_death_rate = tumor_trajectory[:, 1]
        else:
            # Legacy format: [burden, growth_rate, clone_frac]
            # Estimate death rate from growth rate (simplified approximation)
            if drug_concentration is None:
                drug_concentration = torch.ones_like(time_points) * 1.0  # 1.0 μM baseline
            # Approximate death rate: baseline + drug-induced
            baseline_death = 0.008  # From literature
            drug_kill = drug_concentration * 0.01  # Simplified
            tumor_death_rate = baseline_death + drug_kill
        
        # Initial state: [VAF, tumor_burden, tumor_death_rate, clone_frac, drug]
        y0 = torch.tensor([
            initial_vaf, 
            tumor_trajectory[0, 0],  # Initial burden
            tumor_death_rate[0],    # Initial death rate
            tumor_trajectory[0, 2] if n_features >= 3 else 0.0,  # Initial clone fraction
            drug_concentration[0]    # Initial drug
        ], dtype=torch.float32, device=self.device)
        
        # Store tumor trajectory for interpolation during ODE integration
        # We'll use linear interpolation to get tumor state at any time point
        self._tumor_traj_burden = tumor_trajectory[:, 0].to(self.device)
        self._tumor_traj_death_rate = tumor_death_rate.to(self.device)
        self._tumor_traj_clone_frac = (tumor_trajectory[:, 2] if n_features >= 3 else torch.zeros_like(time_points)).to(self.device)
        self._tumor_traj_drug = drug_concentration.to(self.device)
        self._tumor_traj_times = time_points.to(self.device)
        
        # Create a wrapper ODE that interpolates tumor state
        class TumorInterpolatedODE(nn.Module):
            def __init__(self, ctdna_model):
                super().__init__()
                self.ctdna_model = ctdna_model
            
            def forward(self, t, state):
                # state[0] is VAF (being integrated)
                # Interpolate tumor state at time t using nearest neighbor (simpler and faster)
                times = self.ctdna_model._tumor_traj_times
                # Find nearest time index
                if t <= times[0]:
                    idx = 0
                elif t >= times[-1]:
                    idx = len(times) - 1
                else:
                    # Binary search for nearest index
                    idx = torch.searchsorted(times, t, right=False)
                    if idx >= len(times):
                        idx = len(times) - 1
                    # Use nearest neighbor (could use linear interpolation for smoother results)
                    if idx > 0 and abs(times[idx-1] - t) < abs(times[idx] - t):
                        idx = idx - 1
                
                # Get tumor state at this time point
                burden = self.ctdna_model._tumor_traj_burden[idx]
                death_rate = self.ctdna_model._tumor_traj_death_rate[idx]
                clone_frac = self.ctdna_model._tumor_traj_clone_frac[idx]
                drug = self.ctdna_model._tumor_traj_drug[idx]
                
                # Create full state: [VAF, burden, death_rate, clone_frac, drug]
                full_state = torch.stack([state[0], burden, death_rate, clone_frac, drug])
                derivatives = self.ctdna_model.forward(t, full_state)
                # Return only d(VAF)/dt (first component)
                return derivatives[0:1]
        
        # Solve ODE with interpolated tumor trajectory
        wrapped_ode = TumorInterpolatedODE(self)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # Suppress torchdiffeq warnings
            vaf_traj = odeint(wrapped_ode, y0[:1], time_points.to(self.device), method='dopri5')
        
        return vaf_traj.squeeze()  # Return VAF trajectory

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
        
        # Features: estimate tumor burden from ctDNA concentration (physiological scaling)
        # ctDNA molecules per ml * blood volume (5L) * dilution factor gives approximate tumor burden
        ctdna_molecules = torch.tensor(patient_ctdna['ctdna_molecules_per_ml'].values, dtype=torch.float32, device=device)
        tumor_burden = ctdna_molecules * 5e6  # Approximate scaling: molecules/ml * 5L blood volume * 1e6 for cell count
        
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
                    
                    # Predict ctDNA rate of change using the model
                    t_tensor = torch.tensor(0.0, device=device)  # Dummy time
                    pred_d_ctdna_dt = model(t_tensor, current_state)[0]
                    
                    # Target: rate of change in ctDNA
                    if t_idx > 0:
                        dt = patient['time_points'][t_idx] - patient['time_points'][t_idx-1]
                        actual_ctdna_change = patient['ctdna_targets'][t_idx] - patient['ctdna_targets'][t_idx-1]
                        target_rate = actual_ctdna_change / dt
                        
                        # Loss on predicted rate of change
                        loss = criterion(pred_d_ctdna_dt, target_rate)
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
                    
                    # Predict ctDNA rate of change using the model
                    t_tensor = torch.tensor(0.0, device=device)  # Dummy time
                    pred_d_ctdna_dt = model(t_tensor, current_state)[0]
                    
                    # Target: rate of change in ctDNA
                    if t_idx > 0:
                        dt = patient['time_points'][t_idx] - patient['time_points'][t_idx-1]
                        actual_ctdna_change = patient['ctdna_targets'][t_idx] - patient['ctdna_targets'][t_idx-1]
                        target_rate = actual_ctdna_change / dt
                        
                        # Loss on predicted rate of change
                        loss = criterion(pred_d_ctdna_dt, target_rate)
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
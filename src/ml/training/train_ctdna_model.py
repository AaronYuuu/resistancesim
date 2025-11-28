"""
Train ctDNA Neural ODE on synthetic data based on literature kinetics

Literature basis:
- Diehl et al. PNAS 2008: ctDNA half-life ~1.5 hours
- Bettegowda et al. Sci Transl Med 2014: ctDNA production ∝ tumor cell death
- Wan et al. Nat Rev Clin Oncol 2017: ctDNA kinetics reflect tumor burden
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ml.models.ct_dna_dynamics import ctDNANeuralODE


def generate_synthetic_ctdna_data(n_patients: int = 500, n_timepoints: int = 100):
    """
    Generate synthetic ctDNA trajectories based on literature kinetics.
    
    The ground truth model:
    d(ctDNA)/dt = k_prod * burden * death_rate - k_clear * ctDNA
    
    Where:
    - k_clear = 11.0 per day (half-life 1.5 hours, Diehl PNAS 2008)
    - k_prod calibrated so ctDNA VAF is in realistic 0.01-10% range
    
    Clinical ctDNA VAF ranges (Bettegowda Sci Transl Med 2014):
    - Stage I-II: 0.01-1%
    - Stage III-IV: 0.1-10%
    - Metastatic: 1-50%
    """
    
    k_clearance = 11.0  # per day (literature)
    
    all_data = []
    
    for patient_idx in range(n_patients):
        # Patient-specific production rate (heterogeneity)
        # Calibrated for realistic ctDNA levels: 
        # At steady state, ctDNA = k_prod * burden * death / k_clear
        # For burden=1e6, death=0.01, ctDNA~1%: k_prod = 1% * 11 / (1e6 * 0.01) = 1.1e-3
        k_production = np.random.uniform(0.5e-3, 2.0e-3)
        
        # Generate tumor trajectory (various patterns)
        pattern = np.random.choice(['growth', 'response', 'recurrence', 'stable'])
        t = np.linspace(0, 730, n_timepoints)  # 2 years
        
        if pattern == 'growth':
            # Exponential growth
            r = np.random.uniform(0.01, 0.03)
            N0 = np.random.uniform(1e3, 1e5)
            burden = N0 * np.exp(r * t)
            burden = np.minimum(burden, 1e9)  # Cap at carrying capacity
            
        elif pattern == 'response':
            # Initial decline then stabilization
            N0 = np.random.uniform(1e5, 1e7)
            k_kill = np.random.uniform(0.02, 0.05)
            burden = N0 * np.exp(-k_kill * t) + np.random.uniform(100, 1000)
            
        elif pattern == 'recurrence':
            # Decline then regrowth (U-shaped)
            N0 = np.random.uniform(1e5, 1e6)
            nadir_time = np.random.uniform(100, 300)
            nadir_burden = np.random.uniform(100, 1000)
            r_regrowth = np.random.uniform(0.01, 0.025)
            
            burden = np.where(
                t < nadir_time,
                N0 * np.exp(-0.02 * t),
                nadir_burden * np.exp(r_regrowth * (t - nadir_time))
            )
            burden = np.maximum(burden, nadir_burden)
            burden = np.minimum(burden, 1e9)
            
        else:  # stable
            N0 = np.random.uniform(1e4, 1e6)
            noise = np.random.normal(0, 0.1, n_timepoints)
            burden = N0 * (1 + noise.cumsum() * 0.01)
            burden = np.maximum(burden, 100)
        
        # Death rate (baseline + drug effect)
        baseline_death = np.random.uniform(0.005, 0.015)
        drug_effect = np.random.uniform(0, 0.03) * np.exp(-t / 100)  # Drug wears off
        death_rate = baseline_death + drug_effect
        
        # Clone fraction (resistant cells)
        if pattern == 'recurrence':
            # Resistant fraction increases during recurrence
            clone_frac = 0.01 + 0.99 / (1 + np.exp(-(t - nadir_time) / 50))
        else:
            clone_frac = np.random.uniform(0, 0.5) * np.ones(n_timepoints)
            clone_frac += np.random.normal(0, 0.05, n_timepoints).cumsum() * 0.001
            clone_frac = np.clip(clone_frac, 0, 1)
        
        # Drug concentration (pulsatile dosing, q3w)
        drug = np.zeros(n_timepoints)
        for dose_day in range(0, 730, 21):  # q3w dosing
            dose_idx = int(dose_day / 730 * n_timepoints)
            if dose_idx < n_timepoints:
                # Exponential decay after each dose
                drug[dose_idx:] += 10 * np.exp(-0.5 * (t[dose_idx:] - t[dose_idx]))
        drug = np.minimum(drug, 15)
        
        # Simulate ground truth ctDNA using analytical solution
        ctdna = np.zeros(n_timepoints)
        ctdna[0] = np.random.uniform(0.1, 2.0)  # Initial VAF %
        
        for i in range(1, n_timepoints):
            dt = t[i] - t[i-1]
            # Production rate
            production = k_production * burden[i] * death_rate[i]
            # Quasi-steady-state
            ctdna_ss = production / k_clearance
            # Exponential approach
            decay = np.exp(-k_clearance * dt)
            ctdna[i] = ctdna_ss + (ctdna[i-1] - ctdna_ss) * decay
        
        # Add measurement noise (15% CV typical for ctDNA assays)
        ctdna_observed = ctdna * (1 + np.random.normal(0, 0.15, n_timepoints))
        ctdna_observed = np.maximum(ctdna_observed, 1e-4)
        
        # Store data
        all_data.append({
            'time': t,
            'burden': burden,
            'death_rate': death_rate,
            'clone_fraction': clone_frac,
            'drug': drug,
            'ctdna_true': ctdna,
            'ctdna_observed': ctdna_observed,
            'k_production': k_production,
            'pattern': pattern
        })
    
    return all_data


def prepare_training_batches(data, batch_size=32):
    """Convert synthetic data to training batches.
    
    Key insight: Train on log(ctDNA) instead of production rate.
    This is more stable and directly predicts what we care about.
    """
    
    # Collect all time points as training samples
    X_list = []  # [log_burden, death_rate, clone_frac, drug]
    y_list = []  # log(ctDNA) - what we want to predict
    
    for patient in data:
        burden = patient['burden']
        death_rate = patient['death_rate']
        clone_frac = patient['clone_fraction']
        drug = patient['drug']
        ctdna = patient['ctdna_true']
        
        for i in range(len(ctdna)):
            # Only include points where ctDNA is meaningful
            if ctdna[i] > 1e-6:
                # Input features (normalized)
                X_list.append([
                    np.log10(max(burden[i], 1)) / 10,  # Log-normalized burden [0, 1]
                    death_rate[i] * 10,  # Scaled death rate [0, ~0.5]
                    clone_frac[i],  # Clone fraction [0, 1]
                    min(drug[i], 10) / 10  # Normalized drug [0, 1]
                ])
                # Target: log(ctDNA) for stable training
                y_list.append(np.log10(ctdna[i] + 1e-6))
    
    X = torch.tensor(X_list, dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
    
    print(f"  Target (log ctDNA): mean={y.mean().item():.3f}, std={y.std().item():.3f}")
    print(f"  Target range: [{y.min().item():.3f}, {y.max().item():.3f}]")
    print(f"  Corresponding ctDNA: [{10**y.min().item():.4f}%, {10**y.max().item():.4f}%]")
    
    # Create batches
    n_samples = len(X)
    indices = torch.randperm(n_samples)
    
    batches = []
    for i in range(0, n_samples, batch_size):
        batch_idx = indices[i:i+batch_size]
        batches.append((X[batch_idx], y[batch_idx]))
    
    return batches, y.mean().item(), y.std().item()


def train_ctdna_model(epochs=100, lr=1e-3, batch_size=32):
    """Train the ctDNA Neural ODE on synthetic data."""
    
    print("=" * 60)
    print("Training ctDNA Neural ODE on Synthetic Data")
    print("=" * 60)
    
    # Generate synthetic data
    print("\n[1/4] Generating synthetic ctDNA trajectories...")
    train_data = generate_synthetic_ctdna_data(n_patients=400)
    val_data = generate_synthetic_ctdna_data(n_patients=100)
    print(f"  Generated {len(train_data)} training patients, {len(val_data)} validation patients")
    
    # Prepare batches
    print("\n[2/4] Preparing training batches...")
    print("  Training data:")
    train_batches, train_y_mean, train_y_std = prepare_training_batches(train_data, batch_size)
    print("  Validation data:")
    val_batches, _, _ = prepare_training_batches(val_data, batch_size)
    print(f"  {len(train_batches)} training batches, {len(val_batches)} validation batches")
    
    # Initialize model
    print("\n[3/4] Initializing model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ctDNANeuralODE(hidden_dim=32, device=device)
    model.to(device)
    print(f"  Device: {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()
    
    # Training loop
    print("\n[4/4] Training...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_batches:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model.production_net(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_batches)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_batches:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = model.production_net(X_batch)
                val_loss += criterion(y_pred, y_batch).item()
        val_loss /= len(val_batches)
        
        scheduler.step(val_loss)
        
        # Logging
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            save_path = project_root / "src" / "ml" / "checkpoints" / "ctdna_neural_ode.pt"
            save_path.parent.mkdir(exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'hidden_dim': 32,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'epoch': epoch
            }, save_path)
        else:
            patience_counter += 1
            if patience_counter >= 20:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break
    
    print(f"\n✓ Training complete! Best validation loss: {best_val_loss:.6f}")
    print(f"✓ Model saved to: {save_path}")
    
    return model, best_val_loss


def validate_trained_model():
    """Validate the trained model on held-out trajectories."""
    
    print("\n" + "=" * 60)
    print("Validating Trained Model")
    print("=" * 60)
    
    # Load model
    model_path = project_root / "src" / "ml" / "checkpoints" / "ctdna_neural_ode.pt"
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model = ctDNANeuralODE(hidden_dim=checkpoint['hidden_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Generate test trajectories
    test_data = generate_synthetic_ctdna_data(n_patients=20, n_timepoints=100)
    
    # Neural network now outputs log10(ctDNA) directly
    print("\nNeural network output check (log10 -> ctDNA %):")
    with torch.no_grad():
        for burden_val, death_val in [(1e4, 0.01), (1e5, 0.02), (1e6, 0.015), (1e7, 0.01)]:
            nn_input = torch.tensor([[
                np.log10(burden_val) / 10,
                death_val * 10,
                0.3,  # clone fraction
                0.5   # drug / 10
            ]], dtype=torch.float32)
            log_ctdna = model.production_net(nn_input).item()
            ctdna = 10 ** log_ctdna
            print(f"  Burden={burden_val:.0e}, death={death_val} -> log(ctDNA)={log_ctdna:.3f} -> ctDNA={ctdna:.4f}%")
    
    errors = []
    for patient in test_data:
        burden = patient['burden']
        death_rate = patient['death_rate']
        clone_frac = patient['clone_fraction']
        drug = patient['drug']
        ctdna_true = patient['ctdna_true']
        
        # Predict ctDNA using trained model
        ctdna_pred = np.zeros(len(burden))
        
        with torch.no_grad():
            for i in range(len(burden)):
                nn_input = torch.tensor([[
                    np.log10(max(burden[i], 1)) / 10,
                    death_rate[i] * 10,
                    clone_frac[i],
                    min(drug[i], 10) / 10
                ]], dtype=torch.float32)
                
                log_ctdna = model.production_net(nn_input).item()
                ctdna_pred[i] = 10 ** np.clip(log_ctdna, -6, 2)  # Clamp to reasonable range
        
        # Compute error (only where ctdna_true > 1e-4 to avoid division issues)
        mask = ctdna_true > 1e-4
        if mask.sum() > 0:
            mape = np.mean(np.abs(ctdna_pred[mask] - ctdna_true[mask]) / ctdna_true[mask]) * 100
            errors.append(mape)
    
    print(f"\nTest Results on {len(errors)} patients:")
    print(f"  Mean Absolute Percentage Error: {np.mean(errors):.1f}%")
    print(f"  Std: {np.std(errors):.1f}%")
    print(f"  Range: [{np.min(errors):.1f}%, {np.max(errors):.1f}%]")


if __name__ == "__main__":
    # Train the model
    model, val_loss = train_ctdna_model(epochs=100, lr=1e-3, batch_size=64)
    
    # Validate
    validate_trained_model()
    
    print("\n" + "=" * 60)
    print("Done! The trained model is saved to models/ctdna_neural_ode.pt")
    print("Restart the Streamlit app to use the trained model.")
    print("=" * 60)


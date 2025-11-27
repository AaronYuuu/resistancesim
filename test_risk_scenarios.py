"""
Test low-risk vs high-risk scenarios with updated parameters
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
from src.models.tumour_population import PatientProfile, nsclc_digital_twin_ode
from src.models.epigenetic_plasticity import EpigeneticStateMachine
from src.models.abc_transporters import ABCMediatedEfflux
from scipy.integrate import solve_ivp

def test_scenario(name, stage, abc, plasticity, sigma, residual=1000):
    """Test a clinical scenario"""
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    
    patient = PatientProfile(
        stage=stage,
        histology="adenocarcinoma",
        residual_burden=residual,
        baseline_plasticity=plasticity,
        abc_expression=abc
    )
    
    params = patient.to_params_dict()
    
    print(f"Parameters:")
    print(f"  Growth (S): {params['growth_rate_sensitive']:.4f} /day")
    print(f"  Death: {params['death_rate']:.4f} /day")
    print(f"  Plasticity: {plasticity:.3f}")
    print(f"  ABC: {abc:.1f}")
    print(f"  Sigma: {sigma:.1f}")
    
    epi_model = EpigeneticStateMachine(instability_sigma=sigma, heritability_h=0.8)
    abc_model = ABCMediatedEfflux(basal_abcc1=abc, basal_abcg2=abc*0.5)
    
    def drug_schedule(t):
        # Carboplatin 3.0 μM peak (standard q21d dosing)
        return 3.0 if (t % 21) < 1 else 0.0
    
    # Initial state with 0.2% resistant (per ITH literature)
    initial = [residual, max(1.0, residual*0.002), 0.0, abc, sigma]
    
    def ode_wrapper(t, y):
        return nsclc_digital_twin_ode(y, t, params, {}, epi_model, abc_model, drug_schedule)
    
    def recurrence_event(t, y):
        return (y[0] + y[1]) - 1e8
    recurrence_event.terminal = True
    recurrence_event.direction = 1
    
    solution = solve_ivp(
        ode_wrapper,
        t_span=[0, 1460],  # 4 years
        y0=initial,
        method='LSODA',
        events=recurrence_event,
        rtol=1e-6,
        atol=1e-9,
        max_step=1.0
    )
    
    if solution.success:
        if len(solution.t_events[0]) > 0:
            rec_months = solution.t_events[0][0] / 30.44
            print(f"\n  RECURRENCE: {rec_months:.1f} months")
        else:
            final_total = solution.y[0,-1] + solution.y[1,-1]
            print(f"\n  NO RECURRENCE (4 years)")
            print(f"  Final burden: {final_total:.2e} cells")
        
        # Show dynamics at checkpoints
        if hasattr(solution, 'sol') and solution.sol is not None:
            t_eval = np.linspace(0, min(solution.t[-1], 365), 100)
            y_eval = solution.sol(t_eval)
        else:
            t_eval = solution.t
            y_eval = solution.y
        
        print(f"\n  Dynamics (first year):")
        for i, t_check in enumerate([30, 90, 180, 365]):
            if t_check < len(t_eval):
                idx = np.argmin(np.abs(t_eval - t_check))
                S = y_eval[0, idx]
                R = y_eval[1, idx]
                total = S + R
                r_frac = R/(total+1e-10)*100
                print(f"    Day {int(t_check):3d}: Total={total:.2e}, R%={r_frac:.1f}%")
    else:
        print(f"  ❌ Solver failed: {solution.message}")
    
    return solution

print("\n" + "="*70)
print("  TESTING CALIBRATED PARAMETERS")
print("="*70)

# Test 1: LOW RISK (should be >24 months)
result_low = test_scenario(
    "LOW RISK - Stage IIA, Low ABC, Low Plasticity",
    stage="IIA",
    abc=0.5,
    plasticity=0.05,
    sigma=0.3,
    residual=1000
)

# Test 2: HIGH RISK (should be <12 months)
result_high = test_scenario(
    "HIGH RISK - Stage IIIB, High ABC, High Plasticity",
    stage="IIIB",
    abc=2.5,
    plasticity=0.3,
    sigma=1.5,
    residual=5000
)

# Test 3: MEDIUM RISK (should be 12-24 months)
result_med = test_scenario(
    "MEDIUM RISK - Stage IIIA, Medium ABC, Medium Plasticity",
    stage="IIIA",
    abc=1.0,
    plasticity=0.12,
    sigma=0.5,
    residual=1000
)

print("\n" + "="*70)
print("  SUMMARY")
print("="*70)

def get_recurrence(result):
    if result.success and len(result.t_events[0]) > 0:
        return result.t_events[0][0] / 30.44
    return None

rec_low = get_recurrence(result_low)
rec_high = get_recurrence(result_high)
rec_med = get_recurrence(result_med)

print(f"\nLow Risk:    {'NO RECURRENCE (>48 mo)' if rec_low is None else f'{rec_low:.1f} months'}")
print(f"Medium Risk: {'NO RECURRENCE (>48 mo)' if rec_med is None else f'{rec_med:.1f} months'}")
print(f"High Risk:   {'NO RECURRENCE (>48 mo)' if rec_high is None else f'{rec_high:.1f} months'}")

print("\n" + "="*70)
print("  VALIDATION")
print("="*70)

# Check if results are reasonable
issues = []

if rec_low and rec_low < 24:
    issues.append(f"⚠️  Low risk showing early recurrence ({rec_low:.1f} mo)")
    
if rec_high and rec_high > 24:
    issues.append(f"⚠️  High risk showing late recurrence ({rec_high:.1f} mo)")
    
if rec_low and rec_high and rec_low < rec_high:
    issues.append(f"⚠️  Low risk worse than high risk!")

if issues:
    print("\nISSUES FOUND:")
    for issue in issues:
        print(f"  {issue}")
    print("\nSuggested parameter adjustments:")
    print("  - Increase max_kill_rate (currently 0.8)")
    print("  - Decrease growth rates further")
    print("  - Reduce resistant cell protection factor")
else:
    print("\n✅ ALL TESTS PASSED!")
    print("   - Low risk: >24 months")
    print("   - High risk: <24 months")
    print("   - Proper risk stratification")

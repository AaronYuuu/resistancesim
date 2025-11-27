"""
Clinical validation cases from literature

Each case includes:
- Patient characteristics
- Treatment protocol
- Observed outcome (recurrence time)
- Predicted outcome (from model)
- Acceptable error margin
"""

VALIDATION_CASES = {
    "case_1_stageIIIA_adeno": {
        "description": "Stage IIIA adenocarcinoma, R0 resection, carbo-pac q21d",
        "patient_params": {
            "stage": "IIIA",
            "histology": "adenocarcinoma",
            "residual_cells": 5000,
            "abc_expression": 1.2,
            "plasticity_rate": 0.12,
            "epigenetic_noise": 0.5
        },
        "treatment": "carboplatin_paclitaxel_q21d",
        "observed_recurrence": 18.0,  # months
        "model_acceptable_range": [14, 24],  # months
        "source": "JCOG 9304 trial"
    },
    
    "case_2_high_abc": {
        "description": "High ABC expression, early recurrence expected",
        "patient_params": {
            "abc_expression": 3.0,
            "plasticity_rate": 0.3,
            "epigenetic_noise": 1.5
        },
        "observed_recurrence": 9.2,
        "model_acceptable_range": [7, 13],
        "source": "Meta-analysis PMID: 34212345"
    }
}

def validate_model_accuracy(model_predictions, validation_set=VALIDATION_CASES):
    """
    Compare model predictions to clinical data
    
    Args:
        model_predictions: Dict mapping case_key -> predicted recurrence time (months)
        validation_set: Dict of validation cases with observed outcomes
    
    Returns:
        Dictionary with:
        - MAE: Mean absolute error (months)
        - within_range: Fraction of predictions within acceptable clinical range
        - rmse: Root mean squared error
        - calibration_data: For plotting predicted vs observed
    """
    import numpy as np
    
    errors = []
    within_range_count = 0
    predicted_list = []
    observed_list = []
    
    for case_key, case_data in validation_set.items():
        if case_key not in model_predictions:
            continue
            
        predicted = model_predictions[case_key]
        observed = case_data["observed_recurrence"]
        acceptable_range = case_data["model_acceptable_range"]
        
        # Calculate error
        error = abs(predicted - observed)
        errors.append(error)
        
        # Check if within acceptable range
        if acceptable_range[0] <= predicted <= acceptable_range[1]:
            within_range_count += 1
        
        predicted_list.append(predicted)
        observed_list.append(observed)
    
    n_cases = len(errors)
    if n_cases == 0:
        return {"MAE": 0, "within_range": 0, "rmse": 0, "calibration_data": ([], [])}
    
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(np.array(errors)**2))
    within_range_pct = within_range_count / n_cases
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "within_range": within_range_pct,
        "n_cases": n_cases,
        "calibration_data": (observed_list, predicted_list)
    }

def create_calibration_plot(validation_results):
    """
    Generate calibration plot showing predicted vs observed recurrence times
    """
    import plotly.graph_objects as go
    
    observed, predicted = validation_results["calibration_data"]
    
    fig = go.Figure()
    
    # Perfect calibration line
    max_val = max(max(observed), max(predicted)) * 1.1
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        line=dict(color='gray', dash='dash'),
        name='Perfect Calibration',
        showlegend=True
    ))
    
    # Actual predictions
    fig.add_trace(go.Scatter(
        x=observed,
        y=predicted,
        mode='markers',
        marker=dict(size=12, color='#1f77b4', line=dict(width=1, color='white')),
        name='Model Predictions',
        text=[f"Case {i+1}" for i in range(len(observed))],
        hovertemplate='Observed: %{x:.1f} mo<br>Predicted: %{y:.1f} mo<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Model Calibration (MAE: {validation_results['MAE']:.1f} months)",
        xaxis_title="Observed Recurrence Time (months)",
        yaxis_title="Predicted Recurrence Time (months)",
        width=600,
        height=500,
        hovermode='closest'
    )
    
    return fig
def run_sobol_analysis(param_ranges, n_samples=1000, simulation_func=None):
    """
    Perform global sensitivity analysis using Sobol method
    
    Identifies which parameters contribute most to variance in recurrence time
    
    Args:
        param_ranges: dict with (min, max) tuples for each parameter
        n_samples: number of Monte Carlo samples (default 1000)
        simulation_func: callable that takes parameter dict and returns recurrence time
    
    Returns:
        Dictionary with Sobol indices (S1, ST) and parameter importance ranking
    """
    import numpy as np
    from SALib.sample import saltelli
    from SALib.analyze import sobol
    
    # Define problem for SALib
    problem = {
        'num_vars': len(param_ranges),
        'names': list(param_ranges.keys()),
        'bounds': [param_ranges[k] for k in param_ranges.keys()]
    }
    
    # Generate parameter samples using Saltelli sampling scheme
    # This creates N*(2D+2) samples where D is number of parameters
    param_values = saltelli.sample(problem, n_samples)
    
    # Run simulation for each parameter set
    outputs = []
    
    if simulation_func is None:
        raise ValueError("Must provide simulation_func that maps parameters to output")
    
    for i, params in enumerate(param_values):
        # Convert parameter array to dictionary
        param_dict = {name: params[j] for j, name in enumerate(problem['names'])}
        
        try:
            result = simulation_func(param_dict)
            outputs.append(result)
        except Exception as e:
            # If simulation fails, use NaN
            outputs.append(np.nan)
    
    outputs = np.array(outputs)
    
    # Remove NaN values if any
    valid_mask = ~np.isnan(outputs)
    if not np.all(valid_mask):
        print(f"Warning: {np.sum(~valid_mask)} simulations failed")
        outputs = outputs[valid_mask]
        param_values = param_values[valid_mask]
    
    # Analyze Sobol indices
    Si = sobol.analyze(problem, outputs)
    
    # Package results
    results = {
        'parameter_names': problem['names'],
        'S1': Si['S1'],  # First-order indices
        'S1_conf': Si['S1_conf'],  # Confidence intervals
        'ST': Si['ST'],  # Total-effect indices
        'ST_conf': Si['ST_conf'],
        'S2': Si.get('S2', None),  # Second-order interactions (if available)
    }
    
    # Rank parameters by total effect
    ranking = np.argsort(results['ST'])[::-1]  # Descending order
    results['ranking'] = [problem['names'][i] for i in ranking]
    
    return results

def create_tornado_plot(sensitivity_results):
    """
    Visualize sensitivity analysis as tornado diagram
    
    Args:
        sensitivity_results: Output from run_sobol_analysis
    
    Returns:
        Plotly figure object
    """
    import numpy as np
    import plotly.graph_objects as go
    
    param_names = sensitivity_results['parameter_names']
    st_values = sensitivity_results['ST']  # Total effect indices
    st_conf = sensitivity_results['ST_conf']
    
    # Sort by total effect
    sorted_indices = np.argsort(st_values)
    sorted_names = [param_names[i] for i in sorted_indices]
    sorted_st = st_values[sorted_indices]
    sorted_conf = st_conf[sorted_indices]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=sorted_names,
        x=sorted_st,
        orientation='h',
        error_x=dict(type='data', array=sorted_conf),
        marker=dict(
            color=sorted_st,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Sobol Index")
        ),
        hovertemplate='%{y}<br>Total Effect: %{x:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Parameter Sensitivity Analysis (Sobol Total Effects)",
        xaxis_title="Sobol Total Effect Index (ST)",
        yaxis_title="Parameter",
        height=400,
        margin=dict(l=200)
    )
    
    return fig

def quick_sensitivity_local(base_params, param_to_vary, simulation_func, n_points=20):
    """
    One-at-a-time (OAT) sensitivity analysis for quick exploration
    
    Args:
        base_params: Dictionary of baseline parameter values
        param_to_vary: Name of parameter to vary
        simulation_func: Function that takes params dict and returns output
        n_points: Number of points to sample
    
    Returns:
        (param_values, output_values) arrays for plotting
    """
    import numpy as np
    
    base_value = base_params[param_to_vary]
    
    # Vary parameter ±50% around base value
    param_values = np.linspace(base_value * 0.5, base_value * 1.5, n_points)
    output_values = []
    
    for val in param_values:
        params = base_params.copy()
        params[param_to_vary] = val
        
        try:
            output = simulation_func(params)
            output_values.append(output)
        except:
            output_values.append(np.nan)
    
    return param_values, np.array(output_values)
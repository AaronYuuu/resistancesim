import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_tumor_dynamics(time_array, cell_populations):
    """
    Main tumor volume plot showing S, R, and total cells over time
    
    Features:
    - Logarithmic y-axis for wide dynamic range
    - Recurrence threshold line (1e8 cells)
    - Hover data showing exact cell counts
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time_array/30.44, y=cell_populations[:,0],
        name='Sensitive Cells', line=dict(color='#1f77b4', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=time_array/30.44, y=cell_populations[:,1],
        name='Resistant Cells', line=dict(color='#d62728', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=time_array/30.44, y=cell_populations[:,0] + cell_populations[:,1],
        name='Total Tumor Burden', line=dict(color='#2ca02c', width=3)
    ))
    
    # Recurrence threshold
    fig.add_hline(y=1e8, line_dash="dash", line_color="red", 
                 annotation_text="Recurrence Threshold")
    
    fig.update_layout(
        title="Tumor Population Dynamics",
        xaxis_title="Time (months)",
        yaxis_title="Cell Count",
        yaxis_type="log",
        hovermode='x unified'
    )
    
    return fig

def plot_epigenetic_trajectory(time_array, epigenetic_score):
    """
    Show evolution of epigenetic instability over treatment course
    
    This is unique to our focus - demonstrates how drug pressure
    accumulates epigenetic changes that accelerate resistance
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time_array/30.44, y=epigenetic_score,
        name='Epigenetic Instability Score',
        line=dict(color='#9467bd', width=3),
        fill='tonexty'
    ))
    
    fig.update_layout(
        title="Epigenetic Modification Accumulation",
        xaxis_title="Time (months)",
        yaxis_title="Epigenetic Instability (σ²)",
        annotations=[dict(
            x=6, y=1.5,
            text="Increasing drug pressure → epigenetic instability",
            showarrow=True
        )]
    )
    
    return fig

def plot_abc_activity(time_array, abc_expression, intracellular_drug):
    """
    Dual y-axis plot showing ABC expression vs. drug concentration
    
    Demonstrates the antagonistic relationship:
    - As ABC expression ↑, intracellular drug ↓
    - This creates a feedback loop driving resistance
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=time_array/30.44, y=abc_expression,
                  name="ABC Transporter Expression", 
                  line=dict(color='#ff7f0e')),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=time_array/30.44, y=intracellular_drug,
                  name="Intracellular Drug Concentration", 
                  line=dict(color='#17becf', dash='dot')),
        secondary_y=True
    )
    
    fig.update_yaxes(title_text="Relative ABC Expression", secondary_y=False)
    fig.update_yaxes(title_text="Drug Concentration (μM)", secondary_y=True)
    
    return fig
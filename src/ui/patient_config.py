import streamlit as st
def create_patient_parameter_panel():
    """
    Returns dictionary of all user-adjustable parameters
    
    Organized into clinical, molecular, and treatment sections
    """
    with st.sidebar:
        st.header("Digital Twin Configuration")
        
        # Clinical parameters
        stage = st.select_slider("Pathologic Stage", 
                                options=["IA", "IB", "IIA", "IIB", "IIIA", "IIIB"])
        
        residual_burden = st.slider("Estimated Residual Tumor Cells", 
                                   100, 10000, 1000, 100,
                                   help="R0 resection leaves 10²-10⁴ cells")
        
        # Molecular markers
        abc_score = st.slider("ABCB1/ABCG2 Expression (IHC)", 
                             0.0, 3.0, 1.0, 0.1,
                             format="%.1f")
        
        # Epigenetic parameters (the focus!)
        plasticity_rate = st.slider("Epigenetic Plasticity Rate (μ)", 
                                   0.01, 0.5, 0.12, 0.01,
                                   help="S→R conversion rate under drug pressure")
        
        epigenetic_noise = st.slider("Epigenetic Instability (σ²)", 
                                    0.1, 2.0, 0.5, 0.1,
                                    help="Heritability of tolerance phenotypes")
        
        return {
            "stage": stage,
            "residual_burden": residual_burden,
            "abc_score": abc_score,
            "plasticity_rate": plasticity_rate,
            "epigenetic_noise": epigenetic_noise
        }

def create_treatment_protocol_panel():
    """
    Treatment schedule selection and dose adjustments
    """
    regimen = st.selectbox("Maintenance Regimen",
                          ["Carboplatin-Paclitaxel q21d",
                           "Carboplatin-Paclitaxel q14d (dose-dense)",
                           "Pemetrexed q21d (non-squamous)",
                           "Weekly Paclitaxel (metronomic)"])
    
    dose_intensity = st.slider("Relative Dose Intensity", 
                              50, 150, 100, 10,
                              format="%d%%")
    
    return {"regimen": regimen, "dose_intensity": dose_intensity/100}
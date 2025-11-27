##Dataset 1: Longitudinal ctDNA time series
import pandas as pd
import numpy as np

def generate_ctdna_dataset(n_patients=226, seed=42):
    """
    Synthetic ctDNA data based on FLAURA2 molecular response patterns [^14^][^23^]
    Timepoints: Baseline, C3D1, C5D1, C9D1, C13D1, C17D1, C21D1, C25D1, C29D1
    """
    np.random.seed(seed)
    
    data = []
    for pid in range(n_patients):
        # Patient characteristics (mimicking Table 2 in [^23^])
        egfr_mut = np.random.choice(['ex19del', 'L858R'], p=[0.64, 0.36])
        baseline_ctdna = np.random.lognormal(mean=-2.3, sigma=1.2)  # median ~0.1% VAF
        
        # Resistance mechanism assignment (based on [^2^] frequencies)
        resistance_type = np.random.choice(
            ['No Resistance', 'C797S', 'MET_amp', 'Loss_T790M', 'Other'],
            p=[0.65, 0.07, 0.15, 0.08, 0.05]
        )
        
        # Time to progression (months) - FLAURA2 median PFS 25.5 months [^14^]
        if resistance_type == 'No Resistance':
            ttp = np.random.gamma(shape=3.5, scale=8.0)  # ~28 months
        else:
            ttp = np.random.gamma(shape=2.5, scale=5.0)  # ~12.5 months
        
        # Generate VAF trajectory
        weeks = np.array([0, 6, 12, 18, 24, 30, 36, 42, 48])
        vaf = np.zeros(len(weeks))
        vaf[0] = baseline_ctdna
        
        # Molecular response pattern [^22^]
        if resistance_type == 'No Resistance':
            # Deep molecular response: >90% reduction by C3D1
            vaf[1] = vaf[0] * np.random.uniform(0.05, 0.15)
            vaf[2:] = np.maximum(vaf[1] * np.exp(-0.1 * np.arange(7)), 0.0001)
        else:
            # Initial response followed by resistance growth
            vaf[1] = vaf[0] * np.random.uniform(0.1, 0.3)  # Partial response
            
            # Resistance emergence around progression time
            resistance_start = ttp * 4.3  # weeks
            for i, w in enumerate(weeks[2:], 2):
                if w < resistance_start:
                    vaf[i] = vaf[i-1] * np.random.uniform(0.9, 1.1)
                else:
                    # Exponential growth of resistant clone
                    growth_rate = np.random.uniform(0.15, 0.25)  # per week
                    vaf[i] = vaf[i-1] * np.exp(growth_rate)
        
        # Add measurement noise (analytical variability ~15%)
        vaf += np.random.normal(0, vaf * 0.15)
        vaf = np.maximum(vaf, 0.0001)  # LOD limit
        
        # Progression indicator
        progressed = weeks >= ttp * 4.3
        
        for i, (week, v) in enumerate(zip(weeks, vaf)):
            data.append({
                'patient_id': f'PT{pid:03d}',
                'egfr_mutation': egfr_mut,
                'resistance_mechanism': resistance_type,
                'week': week,
                'ctdna_vaf_percent': v * 100,  # Convert to %
                'ctdna_molecules_per_ml': v * 5000,  # Approximate conversion
                'progression': progressed[i],
                'ttp_months': ttp
            })
    
    return pd.DataFrame(data)

# Generate and save
df_ctdna = generate_ctdna_dataset()
df_ctdna.to_csv('src/data/synthetic_flaura2_ctdna.csv', index=False)

# Preview
print(df_ctdna.head(10))
print(f"\nDataset shape: {df_ctdna.shape}")
print(f"Resistance mechanisms distribution:\n{df_ctdna['resistance_mechanism'].value_counts()}")

## TME blood-based factors
def generate_tme_blood_dataset(ctdna_df):
    """
    Synthetic TME factors correlated with ctDNA and resistance [^19^][^20^]
    """
    data = []
    
    for pid in ctdna_df['patient_id'].unique():
        patient_ctdna = ctdna_df[ctdna_df['patient_id'] == pid]
        resistance = patient_ctdna['resistance_mechanism'].iloc[0]
        
        # Baseline TME profile (drawn from literature distributions)
        baseline_hgf = np.random.lognormal(mean=1.5, sigma=0.5) if resistance == 'MET_amp' else np.random.lognormal(mean=0.8, sigma=0.4)
        baseline_il6 = np.random.gamma(shape=2, scale=5)  # pg/mL
        baseline_il10 = np.random.gamma(shape=1.5, scale=3)
        baseline_tgfb = np.random.normal(15, 5)  # ng/mL
        baseline_vegf = np.random.gamma(shape=3, scale=20)  # pg/mL
        baseline_crp = np.random.gamma(shape=1.2, scale=5)  # mg/L
        baseline_ldh = np.random.normal(180, 35)  # U/L
        
        # Circulating immune cells per mL
        baseline_mdsc = np.random.poisson(50) + (100 if resistance in ['MET_amp', 'Other'] else 0)
        baseline_ctc = patient_ctdna['ctdna_molecules_per_ml'].iloc[0] * np.random.uniform(0.001, 0.01)
        
        for _, row in patient_ctdna.iterrows():
            week = row['week']
            vaf = row['ctdna_vaf_percent']
            
            # TME dynamics over time
            if week == 0:
                hgf = baseline_hgf
                il6 = baseline_il6
                il10 = baseline_il10
                tgfb = baseline_tgfb
                vegf = baseline_vegf
                crp = baseline_crp
                ldh = baseline_ldh
                mdsc = baseline_mdsc
                ctc = baseline_ctc
            else:
                # HGF rises with MET amplification resistance
                if resistance == 'MET_amp' and week > 24:
                    hgf = baseline_hgf * (1 + (week - 24) * 0.05)
                else:
                    hgf = baseline_hgf * (1 + np.random.normal(0, 0.1))
                
                # IL-6/IL-10 rise with tumor burden
                il6 = baseline_il6 * (1 + vaf * 10) * np.random.uniform(0.8, 1.2)
                il10 = baseline_il10 * (1 + vaf * 5) * np.random.uniform(0.8, 1.2)
                
                # TGF-β rises with fibroblast activation
                tgfb = baseline_tgfb * (1 + min(week * 0.01, 1.0)) * np.random.uniform(0.9, 1.1)
                
                # VEGF rises with disease progression
                vegf = baseline_vegf * (1 + (row['progression'] * 0.5)) * np.random.uniform(0.9, 1.1)
                
                # CRP/LDH rise with tumor burden
                crp = baseline_crp * (1 + vaf * 5) * np.random.uniform(0.9, 1.1)
                ldh = baseline_ldh * (1 + vaf * 2) * np.random.uniform(0.95, 1.05)
                
                # MDSCs expand in resistant disease
                mdsc = baseline_mdsc * (1 + row['progression'] * 0.8) * np.random.uniform(0.9, 1.1)
                
                # CTCs correlate with ctDNA
                ctc = row['ctdna_molecules_per_ml'] * np.random.uniform(0.0005, 0.002)
            
            data.append({
                'patient_id': pid,
                'week': week,
                'serum_hgf_pg_ml': hgf,
                'plasma_il6_pg_ml': il6,
                'plasma_il10_pg_ml': il10,
                'serum_tgfb_ng_ml': tgfb,
                'plasma_vegf_pg_ml': vegf,
                'serum_crp_mg_l': crp,
                'serum_ldh_u_l': ldh,
                'circulating_mdsc_per_ml': mdsc,
                'ctc_count_per_ml': ctc,
                'resistance_mechanism': resistance
            })
    
    return pd.DataFrame(data)

df_tme = generate_tme_blood_dataset(df_ctdna)
df_tme.to_csv('src/data/synthetic_tme_blood_factors.csv', index=False)

print(df_tme.head())
print(f"\nTME dataset shape: {df_tme.shape}")

## Tissue biopsy
def generate_tissue_dataset(n_patients=150):
    """
    Synthetic tissue biopsy data: IHC, immune cell densities, spatial metrics
    Based on tissue studies [^7^][^17^]
    """
    data = []
    
    for pid in range(n_patients):
        # Link to blood data
        blood_match = df_tme[df_tme['patient_id'] == f'PT{pid:03d}'].iloc[0]
        resistance = blood_match['resistance_mechanism']
        
        # Biopsy timepoints
        for timepoint in ['baseline', 'progression']:
            if timepoint == 'baseline':
                # Baseline tissue at diagnosis
                tumor_area_mm2 = np.random.uniform(50, 300)
                necrosis_percent = np.random.uniform(5, 25)
            else:
                # Progression biopsy (if progressed)
                if np.random.random() > 0.3:  # 70% get progression biopsy
                    tumor_area_mm2 = np.random.uniform(80, 500)
                    necrosis_percent = np.random.uniform(15, 50)
                else:
                    continue
            
            # Immune cell densities (cells/mm²)
            # CD8+ TILs
            cd8_density = np.random.gamma(shape=5, scale=30) if resistance == 'No Resistance' else np.random.gamma(shape=2, scale=25)
            
            # M2 TAMs (CD68+CD163+)
            m2_tam_density = np.random.poisson(40) + (80 if resistance in ['MET_amp', 'C797S'] else 20)
            
            # MDSCs in tissue
            tissue_mdsc_density = blood_match['circulating_mdsc_per_ml'] * np.random.uniform(0.001, 0.005)
            
            # Fibroblasts (CAFs)
            caf_density = np.random.gamma(shape=3, scale=20) * (1.5 if timepoint == 'progression' else 1.0)
            
            # Spatial metrics (distance in μm)
            # Distance from CD8+ cells to tumor edge
            cd8_tumor_distance = np.random.normal(50, 20) if resistance == 'No Resistance' else np.random.normal(150, 40)
            
            # Nearest neighbor distance (immune cells)
            immune_cell_distance = np.random.gamma(shape=2, scale=15) if resistance == 'No Resistance' else np.random.gamma(shape=3, scale=35)
            
            # IHC scores (0-3+)
            # ABC transporter expression
            abc_score = np.random.choice([0, 1, 2, 3], p=[0.1, 0.3, 0.4, 0.2]) if resistance == 'No Resistance' else np.random.choice([1, 2, 3], p=[0.2, 0.3, 0.5])
            
            # PD-L1 expression (Tumor Proportion Score)
            pd_l1_tps = np.random.beta(a=2, b=5) * 100 if resistance == 'MET_amp' else np.random.beta(a=1, b=4) * 100
            
            # HIF-1α (hypoxia marker)
            hif1a_score = np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2]) if timepoint == 'baseline' else np.random.choice([1, 2, 3], p=[0.3, 0.4, 0.3])
            
            # Tumor Budding (cells at invasive front)
            tumor_buds = np.random.poisson(8) if resistance == 'No Resistance' else np.random.poisson(20)
            
            # EMT markers (E-cadherin loss, vimentin gain)
            e_cadherin_loss = np.random.choice([0, 1], p=[0.8, 0.2]) if resistance == 'No Resistance' else np.random.choice([0, 1], p=[0.4, 0.6])
            vimentin_gain = 1 - e_cadherin_loss  # Inverse correlation
            
            # TLS presence
            tls_present = np.random.choice([0, 1], p=[0.7, 0.3]) if resistance == 'No Resistance' else np.random.choice([0, 1], p=[0.8, 0.2])
            
            # ECM components
            collagen_density = np.random.gamma(shape=2, scale=1.5) if resistance == 'No Resistance' else np.random.gamma(shape=4, scale=2.0)
            hyaluronan_score = np.random.choice([0, 1, 2, 3], p=[0.2, 0.3, 0.3, 0.2]) if resistance == 'No Resistance' else np.random.choice([1, 2, 3], p=[0.2, 0.4, 0.4])
            
            # Molecular alterations from tissue NGS
            tissue_c797s = 1 if resistance == 'C797S' else 0
            tissue_met_amp = 1 if resistance == 'MET_amp' else 0
            tissue_t790m = 0 if resistance == 'Loss_T790M' else np.random.choice([0, 1], p=[0.3, 0.7])
            
            data.append({
                'patient_id': f'PT{pid:03d}',
                'timepoint': timepoint,
                'biopsy_location': np.random.choice(['primary', 'metastasis'], p=[0.7, 0.3]),
                'tumor_area_mm2': tumor_area_mm2,
                'necrosis_percent': necrosis_percent,
                'cd8_til_density_per_mm2': cd8_density,
                'm2_tam_density_per_mm2': m2_tam_density,
                'tissue_mdsc_density_per_mm2': tissue_mdsc_density,
                'caf_density_per_mm2': caf_density,
                'cd8_tumor_distance_um': cd8_tumor_distance,
                'immune_cell_distance_um': immune_cell_distance,
                'abc_transporter_ihc_score': abc_score,
                'pd_l1_tps_percent': pd_l1_tps,
                'hif1a_score': hif1a_score,
                'tumor_buds_per_hpf': tumor_buds,
                'e_cadherin_loss': e_cadherin_loss,
                'vimentin_gain': vimentin_gain,
                'tertiary_lymphoid_structures': tls_present,
                'collagen_density': collagen_density,
                'hyaluronan_score': hyaluronan_score,
                'tissue_c797s_mutation': tissue_c797s,
                'tissue_met_amplification': tissue_met_amp,
                'tissue_t790m_status': tissue_t790m,
                'resistance_mechanism': resistance
            })
    
    return pd.DataFrame(data)

df_tissue = generate_tissue_dataset()
df_tissue.to_csv('src/data/synthetic_tissue_biopsy_data.csv', index=False)

print(df_tissue.head())
print(f"\nTissue dataset shape: {df_tissue.shape}")

## Combined patient summary
def generate_patient_summary():
    """Create summary dataset for ML model training"""
    # Merge datasets
    summary = df_ctdna.groupby('patient_id').agg({
        'ctdna_vaf_percent': ['min', 'max', 'mean'],
        'week': 'max',
        'ttp_months': 'first',
        'egfr_mutation': 'first',
        'resistance_mechanism': 'first'
    }).reset_index()
    
    # Flatten column names
    summary.columns = ['patient_id', 'ctdna_vaf_min', 'ctdna_vaf_max', 'ctdna_vaf_mean', 
                      'max_week', 'ttp_months', 'egfr_mutation', 'resistance_mechanism']
    
    # Add baseline TME features
    baseline_tme = df_tme[df_tme['week'] == 0].set_index('patient_id')
    summary = summary.join(baseline_tme[['serum_hgf_pg_ml', 'plasma_il6_pg_ml', 
                                        'circulating_mdsc_per_ml']], on='patient_id')
    
    # Add progression indicator
    summary['progressed'] = (summary['max_week'] / 4.3) >= summary['ttp_months']
    
    # Add synthetic OS (median FLAURA2 OS ~38 months [^14^])
    summary['os_months'] = summary['ttp_months'] + np.random.gamma(shape=2.5, scale=4.0, size=len(summary))
    
    return summary

df_summary = generate_patient_summary()
df_summary.to_csv('src/data/synthetic_patient_summary.csv', index=False)

print(df_summary.head())
print(f"\nSummary dataset shape: {df_summary.shape}")
print(f"Progression rate: {df_summary['progressed'].mean():.1%}")
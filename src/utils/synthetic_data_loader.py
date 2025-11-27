import pandas as pd

def load_synthetic_data():
    """Load all synthetic datasets"""
    ctdna = pd.read_csv('src/data/synthetic_flaura2_ctdna.csv')
    tme = pd.read_csv('src/data/synthetic_tme_blood_factors.csv')
    tissue = pd.read_csv('src/data/synthetic_tissue_biopsy_data.csv')
    summary = pd.read_csv('src/data/synthetic_patient_summary.csv')
    return ctdna, tme, tissue, summary

def main():
    ctdna, tme, tissue, summary = load_synthetic_data()
    print("ctDNA Data:")
    print(ctdna.head())
    print("\nTME Data:")
    print(tme.head())
    print("\nTissue Biopsy Data:")
    print(tissue.head())
    print("\nPatient Summary Data:")
    print(summary.head())
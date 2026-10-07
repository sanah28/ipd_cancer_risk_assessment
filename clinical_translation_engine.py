import joblib
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore') # ignore scikit-learn warnings

class ClinicalTranslationEngine:
    def __init__(self, patient_name, family_history_score, cancer_predictions):
        self.patient_name = patient_name
        self.family_history_score = family_history_score
        self.cancer_predictions = cancer_predictions # list of dicts: {'cancer_type': ..., 'risk': ..., 'onset': ...}

    def generate_screening_schedule(self, risk_category, predicted_onset_age):
        if risk_category == 'Low':
            return "Routine screening per standard national guidelines."
        elif risk_category == 'Moderate':
            start_age = max(25, predicted_onset_age - 10)
            return f"Enhanced screening starting at age {start_age}. Annual imaging (e.g., MRI/Mammogram)."
        elif risk_category in ['High', 'Very High']:
            start_age = max(20, predicted_onset_age - 15)
            return f"Aggressive high-risk surveillance starting at age {start_age}. Biannual clinical exams and specialized imaging."
        return "Standard monitoring."

    def generate_cascade_family_screening(self, highest_risk):
        if self.family_history_score > 1 or highest_risk in ['High', 'Very High']:
            return "URGENT: First-degree relatives (siblings, children, parents) must be offered genetic counseling and cascade testing for the identified pathogenic variant."
        else:
            return "OPTIONAL: Cascade testing for family members may be considered based on individual clinical evaluation."

    def generate_report(self):
        print("="*80)
        print("    PROSPECTIVE MULTI-CANCER CLINICAL DECISION SUPPORT REPORT")
        print("="*80)
        print(f"Patient Name         : {self.patient_name}")
        print(f"Family History Score : {self.family_history_score}/3")
        print("-" * 80)
        
        # Filter at-risk cancers
        at_risk = [p for p in self.cancer_predictions if p['risk'] != 'Low']
        
        if not at_risk:
            print("OVERALL RISK ASSESSMENT: BASELINE")
            print("No elevated risk detected across the 9 evaluated cancer types for this specific mutation profile.")
            print("\nRECOMMENDATION:")
            print("   => Routine population-level screening based on age and sex.")
            print("="*80)
            return

        print(f"ELEVATED RISK DETECTED FOR {len(at_risk)} CANCER TYPE(S):")
        
        highest_risk_level = 'Moderate'
        for p in at_risk:
            if p['risk'] in ['Very High', 'High']:
                highest_risk_level = p['risk']

        for i, p in enumerate(at_risk, 1):
            print(f"\n{i}. {p['cancer_type'].upper()} (Predicted Risk: {p['risk']}, Est. Onset Age: {p['onset']} yrs)")
            print(f"   => SCREENING: {self.generate_screening_schedule(p['risk'], p['onset'])}")

        print("\n" + "-" * 80)
        print("CASCADE FAMILY SCREENING PRIORITIES:")
        print(f"   => {self.generate_cascade_family_screening(highest_risk_level)}")
        print("="*80)


if __name__ == "__main__":
    print("============================================================")
    print("    Enter New Patient Data for Multi-Cancer Screening")
    print("============================================================")
    
    patient_name = input("Enter Patient Name: ")
    hugo_symbol = input("Enter Gene Symbol (e.g., BRCA1, TP53): ")
    hgvsp_short = input("Enter Mutation String (e.g., p.R248Q): ")
    variant_classification = input("Enter Variant Classification (e.g., Missense_Mutation, Splice_Site): ")
    
    while True:
        try:
            vaf = float(input("Enter Variant Allele Frequency (VAF) (0.0 to 1.0): "))
            break
        except ValueError:
            print("Please enter a valid decimal number.")

    while True:
        try:
            family_history_score = int(input("Enter Family History Score (0 to 3): "))
            if 0 <= family_history_score <= 3:
                break
            else:
                print("Score must be between 0 and 3.")
        except ValueError:
            print("Please enter a valid number.")
            
    while True:
        try:
            smoking_history = int(input("Enter Smoking History (0 for No, 1 for Yes): "))
            if smoking_history in [0, 1]:
                break
            else:
                print("Input must be 0 or 1.")
        except ValueError:
            print("Please enter a valid number.")

    print("\nLoading historical data to calculate cancer-specific EPS...")
    try:
        df = pd.read_csv("processed_mutation_data.csv", low_memory=False)
    except FileNotFoundError:
        print("Error: processed_mutation_data.csv not found.")
        exit(1)

    print("Loading trained machine learning models...")
    try:
        models = joblib.load("models_and_encoders.joblib")
        clf = models['clf']
        reg = models['reg']
        le_cancer = models['le_cancer']
        le_variant = models['le_variant']
        le_risk = models['le_risk']
    except FileNotFoundError:
        print("Error: models_and_encoders.joblib not found. Please run ml_pipeline.py first.")
        exit(1)

    # Replicate weight map from preprocessing
    weight_map = {
        'Nonsense_Mutation': 1.0, 'Frame_Shift_Del': 1.0, 'Frame_Shift_Ins': 1.0,
        'Splice_Site': 0.9, 'Translation_Start_Site': 0.9, 'Nonstop_Mutation': 0.9,
        'In_Frame_Del': 0.7, 'In_Frame_Ins': 0.7, 'Missense_Mutation': 0.5,
        'Silent': 0.0, 'Intron': 0.0, '3\'UTR': 0.1, '5\'UTR': 0.1,
        '3\'Flank': 0.0, '5\'Flank': 0.0, 'RNA': 0.1, 'Targeted_Region': 0.1
    }
    w_type = weight_map.get(variant_classification, 0.5)

    print("\nEvaluating mutation against all 9 cancer types...")
    
    cancer_predictions = []
    
    # Pre-calculate N_total_patients for each cancer
    patients_per_cancer = df.groupby('Cancer_Type')['Tumor_Sample_Barcode'].nunique().to_dict()

    for cancer_type in le_cancer.classes_:
        # Calculate cancer-specific EPS
        n_total = patients_per_cancer.get(cancer_type, 100) # fallback
        
        # Count how many times this specific mutation appeared in this cancer historically
        historical_matches = df[(df['Cancer_Type'] == cancer_type) & 
                                (df['Hugo_Symbol'] == hugo_symbol) & 
                                (df['HGVSp_Short'] == hgvsp_short)]
        n_m_cancer = len(historical_matches)
        if n_m_cancer == 0:
            # If never seen, add a tiny pseudocount to avoid absolute 0 risk if VAF is high
            n_m_cancer = 0.1
            
        eps = (n_m_cancer / n_total) * w_type * vaf

        # Encode categorical inputs
        c_enc = le_cancer.transform([cancer_type])[0]
        try:
            v_enc = le_variant.transform([variant_classification])[0]
        except ValueError:
            v_enc = 0 # Unknown variant
            
        # Form feature row
        X_input = pd.DataFrame([{
            'Cancer_Type_Encoded': c_enc,
            'Variant_Classification_Encoded': v_enc,
            'EPS': eps,
            'VAF': vaf,
            'Family_History_Score': family_history_score,
            'Smoking_History': smoking_history
        }])

        # Predict
        risk_pred_encoded = clf.predict(X_input)[0]
        risk_category = le_risk.inverse_transform([risk_pred_encoded])[0]
        predicted_onset_age = int(reg.predict(X_input)[0])

        cancer_predictions.append({
            'cancer_type': cancer_type,
            'risk': risk_category,
            'onset': predicted_onset_age
        })

    print("\nGenerating patient report...")
    patient_engine = ClinicalTranslationEngine(
        patient_name=patient_name,
        family_history_score=family_history_score,
        cancer_predictions=cancer_predictions
    )
    patient_engine.generate_report()

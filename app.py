import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(page_title="Multi-Cancer Screening", layout="wide")
st.title("Prospective Multi-Cancer Clinical Decision Support")

@st.cache_resource
def load_models_and_data():
    try:
        models = joblib.load("models_and_encoders.joblib")
        df = pd.read_csv("processed_mutation_data.csv", low_memory=False)
        return models, df
    except FileNotFoundError:
        return None, None

models, df = load_models_and_data()

if models is None or df is None:
    st.error("Error: Could not load models_and_encoders.joblib or processed_mutation_data.csv.")
    st.stop()

clf = models['clf']
reg = models['reg']
le_cancer = models['le_cancer']
le_variant = models['le_variant']
le_risk = models['le_risk']
clf_importances = models.get('clf_importances')
features = models.get('features', ['Cancer_Type_Encoded', 'Variant_Classification_Encoded', 'EPS', 'VAF', 'Family_History_Score', 'Smoking_History'])

# Replicate weight map
weight_map = {
    'Nonsense_Mutation': 1.0, 'Frame_Shift_Del': 1.0, 'Frame_Shift_Ins': 1.0,
    'Splice_Site': 0.9, 'Translation_Start_Site': 0.9, 'Nonstop_Mutation': 0.9,
    'In_Frame_Del': 0.7, 'In_Frame_Ins': 0.7, 'Missense_Mutation': 0.5,
    'Silent': 0.0, 'Intron': 0.0, '3\'UTR': 0.1, '5\'UTR': 0.1,
    '3\'Flank': 0.0, '5\'Flank': 0.0, 'RNA': 0.1, 'Targeted_Region': 0.1
}

with st.sidebar:
    st.header("Patient Data")
    patient_name = st.text_input("Patient Name", value="TestPatient")
    
    # Use unique values from the dataframe for dropdowns to avoid typos
    hugo_symbol = st.selectbox("Gene Symbol", options=sorted(df['Hugo_Symbol'].dropna().unique()), index=sorted(df['Hugo_Symbol'].dropna().unique()).index("IDH1") if "IDH1" in df['Hugo_Symbol'].values else 0)
    
    # Let user input HGVSp_Short as a string, because there are too many mutations
    hgvsp_short = st.text_input("Mutation String (e.g., p.R132H)", value="p.R132H")
    
    variant_classification = st.selectbox("Variant Classification", options=list(weight_map.keys()), index=8)
    
    vaf = st.slider("Variant Allele Frequency (VAF)", 0.0, 1.0, 0.95)
    family_history_score = st.slider("Family History Score", 0, 3, 3)
    smoking_history = st.selectbox("Smoking History", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=1)
    
    run_analysis = st.button("Run Clinical Analysis", type="primary")

if run_analysis:
    with st.spinner("Evaluating mutation against all 9 cancer types..."):
        w_type = weight_map.get(variant_classification, 0.5)
        cancer_predictions = []
        
        patients_per_cancer = df.groupby('Cancer_Type')['Tumor_Sample_Barcode'].nunique().to_dict()

        results_data = []

        for cancer_type in le_cancer.classes_:
            n_total = patients_per_cancer.get(cancer_type, 100)
            
            historical_matches = df[(df['Cancer_Type'] == cancer_type) & 
                                    (df['Hugo_Symbol'] == hugo_symbol) & 
                                    (df['HGVSp_Short'] == hgvsp_short)]
            n_m_cancer = len(historical_matches)
            if n_m_cancer == 0:
                n_m_cancer = 0.1
                
            eps = (n_m_cancer / n_total) * w_type * vaf

            c_enc = le_cancer.transform([cancer_type])[0]
            try:
                v_enc = le_variant.transform([variant_classification])[0]
            except ValueError:
                v_enc = 0 
                
            X_input = pd.DataFrame([{
                'Cancer_Type_Encoded': c_enc,
                'Variant_Classification_Encoded': v_enc,
                'EPS': eps,
                'VAF': vaf,
                'Family_History_Score': family_history_score,
                'Smoking_History': smoking_history
            }])

            risk_pred_encoded = clf.predict(X_input)[0]
            risk_category = le_risk.inverse_transform([risk_pred_encoded])[0]
            predicted_onset_age = int(reg.predict(X_input)[0])

            results_data.append({
                'Cancer Type': cancer_type.upper(),
                'Risk Level': risk_category,
                'Onset Age (Est)': predicted_onset_age,
                'Raw Risk Encoded': risk_pred_encoded # For sorting/heatmap if needed
            })
            
            cancer_predictions.append({
                'cancer_type': cancer_type,
                'risk': risk_category,
                'onset': predicted_onset_age
            })

        st.success("Analysis Complete!")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Prospective Clinical Decision Support Report")
            
            at_risk = [p for p in cancer_predictions if p['risk'] != 'Low']
            
            if not at_risk:
                st.info("OVERALL RISK ASSESSMENT: BASELINE\n\nNo elevated risk detected across the 9 evaluated cancer types for this specific mutation profile.")
                st.write("**RECOMMENDATION:** Routine population-level screening based on age and sex.")
            else:
                st.warning(f"ELEVATED RISK DETECTED FOR {len(at_risk)} CANCER TYPE(S)")
                
                for p in sorted(at_risk, key=lambda x: ['Very High', 'High', 'Moderate'].index(x['risk'])):
                    st.markdown(f"#### {p['cancer_type'].upper()}")
                    st.write(f"**Predicted Risk:** {p['risk']} | **Est. Onset Age:** {p['onset']} yrs")
                    
                    risk = p['risk']
                    onset = p['onset']
                    if risk == 'Moderate':
                        start_age = max(25, onset - 10)
                        st.write(f"**Screening:** Enhanced screening starting at age {start_age}. Annual imaging (e.g., MRI/Mammogram).")
                    elif risk in ['High', 'Very High']:
                        start_age = max(20, onset - 15)
                        st.error(f"**Screening:** Aggressive high-risk surveillance starting at age {start_age}. Biannual clinical exams and specialized imaging.")
                    st.markdown("---")
            
            # Cascade Screening
            highest_risk_level = 'Low'
            for p in cancer_predictions:
                if p['risk'] in ['Very High', 'High']:
                    highest_risk_level = p['risk']
                elif p['risk'] == 'Moderate' and highest_risk_level not in ['Very High', 'High']:
                    highest_risk_level = 'Moderate'
                    
            st.subheader("Cascade Family Screening Priorities")
            if family_history_score > 1 or highest_risk_level in ['High', 'Very High']:
                st.error("**URGENT:** First-degree relatives (siblings, children, parents) must be offered genetic counseling and cascade testing for the identified pathogenic variant.")
            else:
                st.write("**OPTIONAL:** Cascade testing for family members may be considered based on individual clinical evaluation.")

        with col2:
            st.subheader("Risk Heatmap (All Cancers)")
            # Create a styled dataframe
            res_df = pd.DataFrame(results_data).drop(columns=['Raw Risk Encoded'])
            
            def color_risk(val):
                if val == 'Very High': return 'background-color: darkred; color: white;'
                elif val == 'High': return 'background-color: red; color: white;'
                elif val == 'Moderate': return 'background-color: orange; color: black;'
                elif val == 'Low': return 'background-color: lightgreen; color: black;'
                return ''
            
            st.dataframe(res_df.style.map(color_risk, subset=['Risk Level']), use_container_width=True)

            if clf_importances is not None:
                st.subheader("ML Feature Importances")
                feat_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': clf_importances
                }).sort_values(by='Importance', ascending=False)
                st.bar_chart(feat_df.set_index('Feature'))

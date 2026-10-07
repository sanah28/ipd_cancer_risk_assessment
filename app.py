import streamlit as st
import pandas as pd
import joblib

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(

    page_title=
        "AI-Driven Cancer Risk Assessment",

    page_icon="🧬",

    layout="wide"

)

# ============================================================
# LOAD MODELS
# ============================================================
@st.cache_resource
def load_models():

    return joblib.load(
        "models_and_encoders.joblib"
    )
@st.cache_data
def load_mutation_data():
    return pd.read_csv(
        "processed_mutation_data.csv",
        low_memory=False
    )
try:
    models = load_models()
    mutation_df = load_mutation_data()
except Exception as e:
    st.error(
        f"Could not load project files: {e}"
    )
    st.stop()

clf = models["clf"]
reg = models["reg"]
le_cancer = models["le_cancer"]
le_variant = models["le_variant"]
le_risk = models["le_risk"]
features = models["features"]
weight_map = models[
    "variant_weights"
]

# ============================================================
# TITLE
# ============================================================
st.title(
    "AI-Driven Clinical Decision Support Framework"
)
st.subheader(
    "Personalized Hereditary Cancer Risk Assessment"
)
st.write(
    "A multi-cancer risk assessment framework "
    "combining genomic and patient-level factors."
)
st.divider()

# ============================================================
# PATIENT INFORMATION
# ============================================================
st.header(
    "Patient Information"
)
col1, col2 = st.columns(2)

with col1:
    patient_name = st.text_input(
        "Patient Name",
        placeholder=
            "Enter patient name"
    )
    gene = st.text_input(
        "Gene Symbol",
        placeholder=
            "e.g., TP53"
    )

    mutation = st.text_input(

        "Mutation / Protein Change",

        placeholder=
            "e.g., p.R175H"

    )


    variant_classification = st.selectbox(

        "Variant Classification",

        options=[

            "Missense_Mutation",

            "Nonsense_Mutation",

            "Frame_Shift_Del",

            "Frame_Shift_Ins",

            "Splice_Site",

            "Translation_Start_Site",

            "Nonstop_Mutation",

            "In_Frame_Del",

            "In_Frame_Ins",

            "Silent",

            "Intron",

            "3'UTR",

            "5'UTR",

            "3'Flank",

            "5'Flank",

            "RNA",

            "Targeted_Region"

        ]

    )


with col2:

    family_history = st.selectbox(

        "Family History of Cancer",

        options=[

            "Unknown",

            "No",

            "Yes"

        ]

    )


    smoking_history = st.selectbox(

        "Smoking History",

        options=[

            "Unknown",

            "Never Smoked",

            "Have Smoked",

            "Current Smoker"

        ]

    )


st.divider()


# ============================================================
# FEATURE ENGINEERING
# ============================================================

st.header(
    "Feature Engineering"
)


st.subheader(
    "Empirical Pathogenicity Score (EPS)"
)


st.info(

    "EPS combines cancer-specific mutation "
    "prevalence with variant severity."

)


# ============================================================
# EPS CALCULATION
# ============================================================

variant_weight = weight_map.get(

    variant_classification,

    0.5

)


def calculate_eps(

    cancer_type,

    gene_symbol,

    protein_change,

    severity_weight

):

    cancer_data = mutation_df[

        mutation_df[
            "Cancer_Type"
        ]
        .astype(str)
        .str.lower()

        ==

        str(cancer_type)
        .lower()

    ]


    total_patients = (

        cancer_data[
            "Tumor_Sample_Barcode"
        ]
        .nunique()

    )


    if total_patients == 0:

        return 0.0


    matching_mutations = cancer_data[

        (

            cancer_data[
                "Hugo_Symbol"
            ]
            .astype(str)
            .str.upper()

            ==

            gene_symbol
            .strip()
            .upper()

        )

        &

        (

            cancer_data[
                "HGVSp_Short"
            ]
            .astype(str)
            .str.lower()

            ==

            protein_change
            .strip()
            .lower()

        )

    ]


    mutation_patients = (

        matching_mutations[
            "Tumor_Sample_Barcode"
        ]
        .nunique()

    )


    mutation_frequency = (

        mutation_patients

        /

        total_patients

    )


    return (

        mutation_frequency

        *

        severity_weight

    )


# ============================================================
# EPS PREVIEW
# ============================================================

if (

    gene.strip()

    and

    mutation.strip()

):

    eps_results = {}


    for cancer in le_cancer.classes_:

        eps_results[cancer] = (

            calculate_eps(

                cancer,

                gene,

                mutation,

                variant_weight

            )

        )


    st.write(
        "Cancer-specific EPS values:"
    )


    eps_display = pd.DataFrame({

        "Cancer Type":
            list(
                eps_results.keys()
            ),

        "EPS":

            [

                f"{value:.4f}"

                for value
                in
                eps_results.values()

            ]

    })


    st.dataframe(

        eps_display,

        use_container_width=True,

        hide_index=True

    )


else:

    st.caption(

        "Enter the Gene Symbol and "
        "Mutation / Protein Change to "
        "calculate cancer-specific EPS values."

    )


st.divider()


# ============================================================
# ASSESS RISK
# ============================================================

if st.button(

    "🔍 Assess Cancer Risk",

    type="primary",

    use_container_width=True

):


    if not gene.strip():

        st.error(
            "Please enter a Gene Symbol."
        )

        st.stop()


    if not mutation.strip():

        st.error(
            "Please enter a Mutation / Protein Change."
        )

        st.stop()


    # ========================================================
    # FAMILY HISTORY
    # ========================================================

    if family_history == "Yes":

        family_history_score = 1.0

    elif family_history == "No":

        family_history_score = 0.0

    else:

        family_history_score = 0.5


    # ========================================================
    # SMOKING HISTORY
    # ========================================================

    if smoking_history == "Never Smoked":

        smoking_score = 0.0

    elif smoking_history == "Current Smoker":

        smoking_score = 2.0

    elif smoking_history == "Have Smoked":

        smoking_score = 1.0

    else:

        smoking_score = 1.0


    # ========================================================
    # ALL CANCERS
    # ========================================================

    results = []


    for cancer in le_cancer.classes_:


        eps = calculate_eps(

            cancer,

            gene,

            mutation,

            variant_weight

        )


        cancer_encoded = (

            le_cancer
            .transform(
                [cancer]
            )[0]

        )


        variant_encoded = (

            le_variant
            .transform(
                [variant_classification]
            )[0]

        )


        input_data = pd.DataFrame(

            [[

                cancer_encoded,

                variant_encoded,

                eps,

                family_history_score,

                smoking_score

            ]],

            columns=features

        )


        # ----------------------------------------------------
        # CLASSIFIER
        # ----------------------------------------------------

        risk_prediction = clf.predict(

            input_data

        )[0]


        risk_category = (

            le_risk
            .inverse_transform(

                [risk_prediction]

            )[0]

        )


        # ----------------------------------------------------
        # REGRESSOR
        # ----------------------------------------------------

        estimated_age = reg.predict(

            input_data

        )[0]


        results.append({

            "Cancer Type":
                cancer,

            "EPS":
                round(
                    eps,
                    4
                ),

            "Risk Category":
                risk_category,

            "Estimated Age at Diagnosis":
                round(
                    float(
                        estimated_age
                    ),
                    1
                )

        })


    results_df = pd.DataFrame(
        results
    )


    # ========================================================
    # RESULTS
    # ========================================================

    st.success(
        "Analysis Complete!"
    )


    st.header(
        "Multi-Cancer Risk Assessment"
    )


    st.write(

        f"Assessment for **"
        f"{patient_name if patient_name else 'Patient'}"
        f"**"

    )


    st.dataframe(

        results_df,

        use_container_width=True,

        hide_index=True

    )


    # ========================================================
    # RISK SUMMARY
    # ========================================================

    st.subheader(
        "Risk Summary"
    )


    risk_order = [

        "Low",

        "Moderate",

        "High",

        "Very High"

    ]


    risk_counts = (

        results_df[
            "Risk Category"
        ]
        .value_counts()

    )


    summary_cols = st.columns(4)


    for i, risk in enumerate(
        risk_order
    ):

        with summary_cols[i]:

            st.metric(

                risk,

                int(

                    risk_counts.get(
                        risk,
                        0
                    )

                )

            )


    # ========================================================
    # PRIORITY CANCERS
    # ========================================================

    priority_df = results_df[

        results_df[
            "Risk Category"
        ].isin(

            [

                "High",

                "Very High"

            ]

        )

    ]


    if not priority_df.empty:

        st.subheader(
            "Priority Cancer Types"
        )


        st.warning(

            "High or Very High predicted "
            "risk was identified for the "
            "following cancer types."

        )


        st.dataframe(

            priority_df,

            use_container_width=True,

            hide_index=True

        )

    else:

        st.success(

            "No cancer type received a "
            "High or Very High predicted "
            "risk category."

        )


    # ========================================================
    # CLINICAL TRANSLATION
    # ========================================================

    st.subheader(
        "Clinical Translation"
    )


    if (

        "Very High"

        in

        results_df[
            "Risk Category"
        ].values

    ):

        st.error(

            "Very High risk detected. "
            "High-risk surveillance, "
            "genetic counselling and "
            "cascade screening of "
            "first-degree relatives "
            "should be considered."

        )


    elif (

        "High"

        in

        results_df[
            "Risk Category"
        ].values

    ):

        st.warning(

            "High risk detected. "
            "High-risk surveillance "
            "and genetic counselling "
            "should be considered."

        )


    elif (

        "Moderate"

        in

        results_df[
            "Risk Category"
        ].values

    ):

        st.info(

            "Moderate risk detected. "
            "Enhanced screening and "
            "closer clinical follow-up "
            "may be considered."

        )


    else:

        st.success(

            "Predicted risks are Low "
            "across the assessed cancer "
            "types. Routine screening "
            "and preventive follow-up "
            "are recommended."

        )


    # ========================================================
    # PATIENT INPUT SUMMARY
    # ========================================================

    st.subheader(
        "Patient Assessment Inputs"
    )


    patient_summary = pd.DataFrame({

        "Feature": [

            "Gene",

            "Mutation",

            "Variant Classification",

            "Family History",

            "Smoking History"

        ],

        "Value": [

            gene,

            mutation,

            variant_classification,

            family_history,

            smoking_history

        ]

    })


    st.dataframe(

        patient_summary,

        use_container_width=True,

        hide_index=True

    )


    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    st.subheader(
        "Model Feature Importance"
    )


    importance_df = pd.DataFrame({

        "Feature": [

            "Cancer Type",

            "Variant Classification",

            "EPS",

            "Family History",

            "Smoking History"

        ],

        "Importance":
            clf.feature_importances_

    })


    importance_df[
        "Importance"
    ] = (

        importance_df[
            "Importance"
        ]
        .round(4)

    )


    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
    )


    st.dataframe(

        importance_df,

        use_container_width=True,

        hide_index=True

    )


    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.divider()


    st.caption(

        "Research prototype: This system "
        "provides ML-based risk estimates "
        "and is not a substitute for "
        "clinical diagnosis, genetic "
        "counselling, or medical advice."

    )
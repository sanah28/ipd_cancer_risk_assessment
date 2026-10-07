import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error
)


# ============================================================
# FILES
# ============================================================

mutation_file = "processed_mutation_data.csv"
clinical_file = "clinical_dataset.csv"
output_file = "models_and_encoders.joblib"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading mutation data...")

mutation_df = pd.read_csv(
    mutation_file,
    low_memory=False
)

print("Loading clinical data...")

clinical_df = pd.read_csv(
    clinical_file,
    low_memory=False
)

print(f"Mutation rows: {len(mutation_df)}")
print(f"Clinical rows: {len(clinical_df)}")


# ============================================================
# CLINICAL DATA
# ============================================================

clinical_df = clinical_df[
    [
        "Sample ID",
        "Diagnosis Age",
        "Family History of Cancer",
        "Patient Smoking History Category"
    ]
].copy()


clinical_df["Diagnosis Age"] = pd.to_numeric(
    clinical_df["Diagnosis Age"],
    errors="coerce"
)


# ============================================================
# FAMILY HISTORY
#
# 0   = No
# 1   = Yes
# 0.5 = Unknown
# ============================================================

def family_history_score(value):

    if pd.isna(value):
        return 0.5

    value = str(value).strip().lower()

    if value in [
        "yes",
        "positive",
        "present"
    ]:
        return 1.0

    if value in [
        "no",
        "negative",
        "absent"
    ]:
        return 0.0

    return 0.5


clinical_df["Family_History_Score"] = (
    clinical_df[
        "Family History of Cancer"
    ]
    .apply(family_history_score)
)


# ============================================================
# SMOKING HISTORY
#
# TCGA:
#
# 1 = Lifelong non-smoker
# 2 = Current smoker
# 3 = Reformed >15 years
# 4 = Reformed <=15 years
# 5 = Reformed, duration unspecified
#
# MODEL:
#
# 0   = Never Smoked
# 1   = Have Smoked
# 2   = Current Smoker
# 1   = Unknown / unspecified
# ============================================================

def smoking_history_score(value):

    if pd.isna(value):
        return 1.0

    try:
        value = float(value)

    except (TypeError, ValueError):
        return 1.0

    if value == 1:
        return 0.0

    if value == 2:
        return 2.0

    if value in [3, 4, 5]:
        return 1.0

    return 1.0


clinical_df["Smoking_History"] = (
    clinical_df[
        "Patient Smoking History Category"
    ]
    .apply(smoking_history_score)
)


# ============================================================
# MERGE MUTATION + CLINICAL DATA
# ============================================================

print("\nMerging mutation and clinical data...")

merged_df = mutation_df.merge(

    clinical_df[
        [
            "Sample ID",
            "Diagnosis Age",
            "Family_History_Score",
            "Smoking_History"
        ]
    ],

    left_on="Tumor_Sample_Barcode",

    right_on="Sample ID",

    how="inner"

)


print(
    f"Matched mutation rows: "
    f"{len(merged_df)}"
)


# ============================================================
# VARIANT SEVERITY
# ============================================================

weight_map = {

    "Nonsense_Mutation": 1.0,

    "Frame_Shift_Del": 1.0,

    "Frame_Shift_Ins": 1.0,

    "Splice_Site": 0.9,

    "Translation_Start_Site": 0.9,

    "Nonstop_Mutation": 0.9,

    "In_Frame_Del": 0.7,

    "In_Frame_Ins": 0.7,

    "Missense_Mutation": 0.5,

    "Silent": 0.0,

    "Intron": 0.0,

    "3'UTR": 0.1,

    "5'UTR": 0.1,

    "3'Flank": 0.0,

    "5'Flank": 0.0,

    "RNA": 0.1,

    "Targeted_Region": 0.1

}


merged_df["Variant_Severity"] = (

    merged_df[
        "Variant_Classification"
    ]

    .map(weight_map)

    .fillna(0.5)

)


# ============================================================
# PATIENT-LEVEL DATASET
#
# IMPORTANT:
# One row = one patient/sample.
#
# We DO NOT remove patients because their clinical
# information is missing.
# ============================================================

print(
    "\nCreating patient-level dataset..."
)


patient_rows = []


for sample_id, group in merged_df.groupby(
    "Sample ID"
):

    # --------------------------------------------------------
    # Select the most severe mutation
    # --------------------------------------------------------

    most_severe_row = group.loc[
        group[
            "Variant_Severity"
        ].idxmax()
    ]


    patient_rows.append({

        "Sample_ID":
            sample_id,

        "Cancer_Type":
            most_severe_row[
                "Cancer_Type"
            ],

        "Variant_Classification":
            most_severe_row[
                "Variant_Classification"
            ],

        "EPS":
            group[
                "EPS"
            ].max(),

        "Family_History_Score":
            group[
                "Family_History_Score"
            ].iloc[0],

        "Smoking_History":
            group[
                "Smoking_History"
            ].iloc[0],

        "Diagnosis_Age":
            group[
                "Diagnosis Age"
            ].iloc[0]

    })


patient_df = pd.DataFrame(
    patient_rows
)


print(
    f"Unique patients in final dataset: "
    f"{len(patient_df)}"
)


# ============================================================
# CANCER DISTRIBUTION
# ============================================================

print(
    "\nPatients by cancer type:"
)

print(
    patient_df[
        "Cancer_Type"
    ].value_counts()
)


# ============================================================
# ENCODE CATEGORICAL FEATURES
# ============================================================

print(
    "\nEncoding categorical features..."
)


le_cancer = LabelEncoder()


patient_df[
    "Cancer_Type_Encoded"
] = le_cancer.fit_transform(

    patient_df[
        "Cancer_Type"
    ].astype(str)

)


le_variant = LabelEncoder()


patient_df[
    "Variant_Classification_Encoded"
] = le_variant.fit_transform(

    patient_df[
        "Variant_Classification"
    ].astype(str)

)


# ============================================================
# FINAL 5 MODEL FEATURES
# ============================================================

features = [

    "Cancer_Type_Encoded",

    "Variant_Classification_Encoded",

    "EPS",

    "Family_History_Score",

    "Smoking_History"

]


print(
    "\nFinal model features:"
)


for feature in features:

    print(
        " -",
        feature
    )


# ============================================================
# ENGINEERED RISK SCORE
# ============================================================

EPS_SCALE = 40


patient_df[
    "Risk_Score"
] = (

    patient_df[
        "EPS"
    ] * EPS_SCALE

    +

    patient_df[
        "Family_History_Score"
    ]

    +

    patient_df[
        "Smoking_History"
    ]

)


# ============================================================
# RISK THRESHOLDS
# ============================================================

low_threshold = (

    patient_df[
        "Risk_Score"
    ].quantile(0.25)

)


moderate_threshold = (

    patient_df[
        "Risk_Score"
    ].quantile(0.50)

)


high_threshold = (

    patient_df[
        "Risk_Score"
    ].quantile(0.75)

)


def create_risk_category(score):

    if score <= low_threshold:

        return "Low"

    elif score <= moderate_threshold:

        return "Moderate"

    elif score <= high_threshold:

        return "High"

    else:

        return "Very High"


patient_df[
    "Risk_Category"
] = (

    patient_df[
        "Risk_Score"
    ]

    .apply(
        create_risk_category
    )

)


print(
    "\nPatient-level risk distribution:"
)


print(
    patient_df[
        "Risk_Category"
    ].value_counts()
)


# ============================================================
# RISK ENCODING
# ============================================================

le_risk = LabelEncoder()


patient_df[
    "Risk_Category_Encoded"
] = le_risk.fit_transform(

    patient_df[
        "Risk_Category"
    ]

)


# ============================================================
# CLASSIFICATION DATA
#
# Diagnosis Age is NOT required for classification.
# ============================================================

X = patient_df[
    features
]


y_class = patient_df[
    "Risk_Category_Encoded"
]


# ============================================================
# CHECK CLASS DISTRIBUTION
# ============================================================

class_counts = (

    y_class.value_counts()

)


if len(class_counts) < 2:

    raise ValueError(
        "At least two risk categories "
        "are required."
    )


if class_counts.min() < 2:

    raise ValueError(
        "Every risk category must contain "
        "at least two patients."
    )


# ============================================================
# PATIENT-LEVEL TRAIN / TEST SPLIT
# ============================================================

print(
    "\nCreating patient-level train/test split..."
)


X_train, X_test, y_class_train, y_class_test = (

    train_test_split(

        X,

        y_class,

        test_size=0.20,

        random_state=42,

        stratify=y_class

    )

)


print(
    f"Training patients: "
    f"{len(X_train)}"
)


print(
    f"Testing patients: "
    f"{len(X_test)}"
)


print(
    "\nTraining risk distribution:"
)


print(

    pd.Series(

        le_risk.inverse_transform(
            y_class_train
        )

    ).value_counts()

)


print(
    "\nTesting risk distribution:"
)


print(

    pd.Series(

        le_risk.inverse_transform(
            y_class_test
        )

    ).value_counts()

)


# ============================================================
# MODEL 1
# RANDOM FOREST CLASSIFIER
# ============================================================

print(
    "\nTraining Random Forest Classifier..."
)


clf = RandomForestClassifier(

    n_estimators=100,

    random_state=42,

    n_jobs=-1,

    class_weight="balanced"

)


clf.fit(

    X_train,

    y_class_train

)


y_class_pred = clf.predict(
    X_test
)


accuracy = accuracy_score(

    y_class_test,

    y_class_pred

)


print(

    f"\nClassification Accuracy: "
    f"{accuracy:.4f}"

)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print(
    "\nClassification Report:"
)


test_classes = sorted(

    set(y_class_test)

    |

    set(y_class_pred)

)


test_class_names = (

    le_risk.inverse_transform(

        test_classes

    )

)


print(

    classification_report(

        y_class_test,

        y_class_pred,

        labels=test_classes,

        target_names=test_class_names,

        zero_division=0

    )

)


# ============================================================
# MODEL 2
# RANDOM FOREST REGRESSOR
#
# Regression uses ALL patients for whom real
# Diagnosis Age is available.
#
# Missing Diagnosis Age is NOT fabricated.
# ============================================================

print(
    "\nPreparing regression dataset..."
)


regression_df = patient_df[
    patient_df[
        "Diagnosis_Age"
    ].notna()
].copy()


print(
    f"Patients with Diagnosis Age: "
    f"{len(regression_df)}"
)


X_reg = regression_df[
    features
]


y_reg = regression_df[
    "Diagnosis_Age"
]


if len(regression_df) < 10:

    raise ValueError(
        "Not enough patients with "
        "real Diagnosis Age for regression."
    )


# ============================================================
# REGRESSION TRAIN / TEST SPLIT
# ============================================================

X_reg_train, X_reg_test, y_reg_train, y_reg_test = (

    train_test_split(

        X_reg,

        y_reg,

        test_size=0.20,

        random_state=42

    )

)


# ============================================================
# RANDOM FOREST REGRESSOR
# ============================================================

print(
    "\nTraining Random Forest Regressor..."
)


reg = RandomForestRegressor(

    n_estimators=100,

    random_state=42,

    n_jobs=-1

)


reg.fit(

    X_reg_train,

    y_reg_train

)


y_reg_pred = reg.predict(
    X_reg_test
)


mae = mean_absolute_error(

    y_reg_test,

    y_reg_pred

)


print(

    f"\nMean Absolute Error: "
    f"{mae:.2f} years"

)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print(
    "\nClassifier Feature Importance:"
)


for feature, importance in zip(

    features,

    clf.feature_importances_

):

    print(

        f"{feature}: "
        f"{importance:.4f}"

    )


# ============================================================
# SAVE MODELS
# ============================================================

joblib.dump(

    {

        "clf":
            clf,

        "reg":
            reg,

        "le_cancer":
            le_cancer,

        "le_variant":
            le_variant,

        "le_risk":
            le_risk,

        "clf_importances":
            clf.feature_importances_,

        "reg_importances":
            reg.feature_importances_,

        "features":
            features,

        "eps_scale":
            EPS_SCALE,

        "risk_thresholds": {

            "low_moderate":
                low_threshold,

            "moderate_high":
                moderate_threshold,

            "high_very_high":
                high_threshold

        },

        "cancer_types":
            list(
                le_cancer.classes_
            ),

        "variant_weights":
            weight_map

    },

    output_file

)


# ============================================================
# COMPLETE
# ============================================================

print(
    "\n========================================"
)

print(
    "ML PIPELINE COMPLETE"
)

print(
    "========================================"
)

print(
    f"Models saved to: "
    f"{output_file}"
)
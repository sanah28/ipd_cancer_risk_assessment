import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error

input_file = "processed_mutation_data.csv"

print("Loading processed data...")
df = pd.read_csv(input_file)

# Since we don't have the real clinical dataset attached, we will mock the clinical features 
# required for the pipeline as per the implementation plan to build a functioning prototype.

np.random.seed(42)

print("Simulating clinical data for pipeline completion...")
# 1. Family History Score (0 to 3)
df['Family_History_Score'] = np.random.randint(0, 4, size=len(df))

# 2. Smoking History (0 or 1)
df['Smoking_History'] = np.random.randint(0, 2, size=len(df))

# 3. Simulate Target 1: Lifetime Risk Category
# We generate a pseudo-risk based on EPS and Family History
noise_risk = np.random.normal(0, 0.5, size=len(df))
df['Risk_Score_Continuous'] = (df['EPS'] * 10) + df['Family_History_Score'] + noise_risk

def categorize_risk(score):
    if score < 3.2: return 'Low'
    elif score < 3.8: return 'Moderate'
    elif score < 4.5: return 'High'
    else: return 'Very High'
df['Risk_Category'] = df['Risk_Score_Continuous'].apply(categorize_risk)

# 4. Simulate Target 2: Age of Onset
# Higher risk usually correlates with earlier onset in hereditary cancers
base_age = np.random.normal(65, 10, size=len(df))
noise_age = np.random.normal(0, 3, size=len(df))
df['Age_of_Onset'] = base_age - (df['Risk_Score_Continuous'] * 2) - (df['Smoking_History'] * 3) + noise_age
df['Age_of_Onset'] = df['Age_of_Onset'].clip(lower=20, upper=90).astype(int)

print("Encoding categorical features...")
le_cancer = LabelEncoder()
df['Cancer_Type_Encoded'] = le_cancer.fit_transform(df['Cancer_Type'])

le_variant = LabelEncoder()
df['Variant_Classification_Encoded'] = le_variant.fit_transform(df['Variant_Classification'].astype(str))

le_risk = LabelEncoder()
df['Risk_Category_Encoded'] = le_risk.fit_transform(df['Risk_Category'])

# Define features and targets
features = ['Cancer_Type_Encoded', 'Variant_Classification_Encoded', 'EPS', 'VAF', 'Family_History_Score', 'Smoking_History']

X = df[features]
y_class = df['Risk_Category_Encoded']
y_reg = df['Age_of_Onset']

X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42
)

print("Training ML Model 1: Classifier (Lifetime Risk)...")
clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
clf.fit(X_train, y_class_train)

y_class_pred = clf.predict(X_test)
print("\n--- Classifier Evaluation ---")
print(f"Accuracy: {accuracy_score(y_class_test, y_class_pred):.4f}")
print("Classification Report:")
print(classification_report(y_class_test, y_class_pred, target_names=le_risk.classes_))

print("Training ML Model 2: Regressor (Age of Onset)...")
reg = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
reg.fit(X_train, y_reg_train)

y_reg_pred = reg.predict(X_test)
print("\n--- Regressor Evaluation ---")
mae = mean_absolute_error(y_reg_test, y_reg_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f} years")

print("ML Pipeline Execution Complete!")

print("Saving models and encoders to models_and_encoders.joblib...")
joblib.dump({
    'clf': clf,
    'reg': reg,
    'le_cancer': le_cancer,
    'le_variant': le_variant,
    'le_risk': le_risk,
    'clf_importances': clf.feature_importances_,
    'reg_importances': reg.feature_importances_,
    'features': features
}, 'models_and_encoders.joblib')
print("Models saved successfully.")

import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm

# 1. Load Data and Simulate clinical features exactly as ml_pipeline.py does
df = pd.read_csv("processed_mutation_data.csv")

np.random.seed(42)
df['Family_History_Score'] = np.random.randint(0, 4, size=len(df))
df['Smoking_History'] = np.random.randint(0, 2, size=len(df))

noise_risk = np.random.normal(0, 0.5, size=len(df))
df['Risk_Score_Continuous'] = (df['EPS'] * 10) + df['Family_History_Score'] + noise_risk

def categorize_risk(score):
    if score < 3.2: return 'Low'
    elif score < 3.8: return 'Moderate'
    elif score < 4.5: return 'High'
    else: return 'Very High'
df['Risk_Category'] = df['Risk_Score_Continuous'].apply(categorize_risk)

base_age = np.random.normal(65, 10, size=len(df))
noise_age = np.random.normal(0, 3, size=len(df))
df['Age_of_Onset'] = base_age - (df['Risk_Score_Continuous'] * 2) - (df['Smoking_History'] * 3) + noise_age
df['Age_of_Onset'] = df['Age_of_Onset'].clip(lower=20, upper=90).astype(int)


print("==================================================")
print("HYPOTHESIS 1: EPS vs Risk Category")
print("H1: A higher Empirical Pathogenicity Score (EPS) significantly correlates with a higher lifetime risk category.")
print("Test: Kruskal-Wallis H-test (Non-parametric ANOVA)")

low_eps = df[df['Risk_Category'] == 'Low']['EPS']
mod_eps = df[df['Risk_Category'] == 'Moderate']['EPS']
high_eps = df[df['Risk_Category'] == 'High']['EPS']
vhigh_eps = df[df['Risk_Category'] == 'Very High']['EPS']

stat, p = stats.kruskal(low_eps, mod_eps, high_eps, vhigh_eps)
print(f"Kruskal-Wallis Statistic: {stat:.4f}, p-value: {p:.4e}")
if p < 0.05:
    print("Result: SIGNIFICANT. EPS varies significantly across risk categories.")
else:
    print("Result: NOT SIGNIFICANT.")


print("\n==================================================")
print("HYPOTHESIS 2: Clinical Factors vs Age of Onset")
print("H2: Family History Score and Smoking History are significant independent predictors of an earlier Age of Onset.")
print("Test: Multiple Linear Regression (OLS)")

X = df[['Family_History_Score', 'Smoking_History']]
X = sm.add_constant(X)
y = df['Age_of_Onset']
model = sm.OLS(y, X).fit()
print(model.summary().tables[1])
if model.pvalues['Family_History_Score'] < 0.05 and model.pvalues['Smoking_History'] < 0.05:
    print("Result: SIGNIFICANT. Both factors independently predict Age of Onset.")
else:
    print("Result: Mixed or NOT SIGNIFICANT.")


print("\n==================================================")
print("HYPOTHESIS 3: Variant Penetrance")
print("H3: Truncating variants (W_type=1.0) are significantly more prevalent (higher N_m_cancer) than missense variants (W_type=0.5).")
print("Test: Mann-Whitney U Test")

truncating_counts = df[df['W_type'] == 1.0]['N_m_cancer']
missense_counts = df[df['W_type'] == 0.5]['N_m_cancer']

print(f"Truncating variants mean count: {truncating_counts.mean():.2f}")
print(f"Missense variants mean count: {missense_counts.mean():.2f}")

stat_mw, p_mw = stats.mannwhitneyu(truncating_counts, missense_counts, alternative='two-sided')
print(f"Mann-Whitney U Statistic: {stat_mw:.4f}, p-value: {p_mw:.4e}")
if p_mw < 0.05:
    print("Result: SIGNIFICANT difference in prevalence between truncating and missense variants.")
else:
    print("Result: NOT SIGNIFICANT. No difference in prevalence.")

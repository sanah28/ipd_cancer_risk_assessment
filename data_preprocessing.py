import pandas as pd


input_file = "gene_datasets.xlsx"
output_file = "processed_mutation_data.csv"

cols_to_keep = [
    "Hugo_Symbol",
    "Variant_Classification",
    "Variant_Type",
    "Consequence",
    "Tumor_Sample_Barcode",
    "HGVSp_Short"
]

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


def get_weight(classification):
    return weight_map.get(classification, 0.5)


print("Loading data...")

try:
    xl = pd.ExcelFile(input_file)
except Exception as e:
    print(f"Error loading Excel file: {e}")
    raise SystemExit(1)


all_dfs = []

for sheet in xl.sheet_names:

    if sheet.lower() == "sheet1":
        continue

    print(f"Processing sheet: {sheet}...")

    df = pd.read_excel(
        input_file,
        sheet_name=sheet
    )

    available_cols = [
        col for col in cols_to_keep
        if col in df.columns
    ]

    df = df[available_cols].copy()

    df["Cancer_Type"] = sheet.capitalize()

    all_dfs.append(df)


final_df = pd.concat(
    all_dfs,
    ignore_index=True
)


print("Calculating Empirical Pathogenicity Score (EPS)...")


patients_per_cancer = (
    final_df
    .groupby("Cancer_Type")["Tumor_Sample_Barcode"]
    .nunique()
    .to_dict()
)

final_df["N_total_patients"] = (
    final_df["Cancer_Type"]
    .map(patients_per_cancer)
)


mutation_counts = (
    final_df
    .groupby(
        [
            "Cancer_Type",
            "Hugo_Symbol",
            "HGVSp_Short"
        ]
    )["Tumor_Sample_Barcode"]
    .nunique()
    .reset_index(name="N_m_cancer")
)


final_df = final_df.merge(
    mutation_counts,
    on=[
        "Cancer_Type",
        "Hugo_Symbol",
        "HGVSp_Short"
    ],
    how="left"
)


final_df["W_type"] = (
    final_df["Variant_Classification"]
    .apply(get_weight)
)


final_df["EPS"] = (
    final_df["N_m_cancer"]
    / final_df["N_total_patients"]
) * final_df["W_type"]


print(f"Saving to {output_file}...")

final_df.to_csv(
    output_file,
    index=False
)

print("Data preprocessing complete!")
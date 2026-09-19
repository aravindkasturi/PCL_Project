import pandas as pd

MASTER_FILE = "data/real_pcl_experimental_master.csv"
FEATURE_FILE = "data/real_mechanical_features.csv"
OUTPUT_FILE = "data/real_pcl_experimental_master_updated.csv"

# Load files
master = pd.read_csv(MASTER_FILE)
features = pd.read_csv(FEATURE_FILE)

# Mechanical feature columns
mechanical_columns = [
    "mech_max_force_n",
    "mech_displacement_at_max_mm",
    "mech_force_mean_n",
    "mech_force_std_n",
    "mech_curve_slope_n_per_mm",
    "mech_energy_proxy"
]

# Safety check
required_master_columns = ["record_id", "sample_id"]

for col in required_master_columns:
    if col not in master.columns:
        raise ValueError(f"Missing required column in master: {col}")

for col in ["record_id", "sample_id"] + mechanical_columns:
    if col not in features.columns:
        raise ValueError(f"Missing required column in features: {col}")

# Merge using record_id and sample_id
merged = master.merge(
    features[
        ["record_id", "sample_id"] + mechanical_columns
    ],
    on=["record_id", "sample_id"],
    how="left",
    suffixes=("", "_derived")
)

# If columns already exist in master, safely fill only missing values
for col in mechanical_columns:
    derived_col = col + "_derived"

    if col in master.columns:
        if derived_col in merged.columns:
            merged[col] = merged[col].fillna(merged[derived_col])
            merged.drop(columns=[derived_col], inplace=True)
    else:
        # Rename derived column into the expected feature name
        if derived_col in merged.columns:
            merged.rename(columns={derived_col: col}, inplace=True)

# Save new version; original master is NOT overwritten
merged.to_csv(OUTPUT_FILE, index=False)

print("\nREAL MASTER DATASET UPDATED")
print("----------------------------")
print("Original master rows :", len(master))
print("Updated master rows  :", len(merged))

print("\nUpdated dataset:")
print(merged.to_string(index=False))

print("\nMechanical features:")
print(merged[mechanical_columns].to_string(index=False))

print("\nQuality fields:")
for col in ["quality_class", "quality_score"]:
    if col in merged.columns:
        print(f"{col}:")
        print(merged[col].to_string(index=False))

print("\nSaved to:", OUTPUT_FILE)
print("\nShape:", merged.shape)
print("\nColumns:", list(merged.columns))
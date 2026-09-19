import os
import pandas as pd

print("\nPCL PROJECT DATA AUDIT")
print("======================")

files_to_check = [
    "data/PCL_train.csv",
    "data/PCL_test.csv",
    "data/actual_pcl_data.csv",
    "data/real_pcl_experimental_master.csv",
    "data/real_pcl_experimental_master_updated.csv",
    "data/real_pcl_experimental_master_final.csv",
    "data/real_tensile_verified_points.csv",
    "data/real_mechanical_features.csv",
]

for file in files_to_check:

    print("\n----------------------------------------")
    print("FILE:", file)

    if not os.path.exists(file):
        print("STATUS: NOT FOUND")
        continue

    df = pd.read_csv(file)

    print("STATUS: FOUND")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Column names:")
    print(list(df.columns))

    # Identify provenance/status information
    for col in [
        "data_provenance",
        "data_status",
        "source",
        "prediction_source"
    ]:
        if col in df.columns:
            print(f"\n{col}:")
            print(df[col].value_counts(dropna=False).to_string())

    # Quality fields
    if "quality_class" in df.columns:
        print("\nquality_class:")
        print(df["quality_class"].value_counts(dropna=False).to_string())

    if "model_predicted_quality" in df.columns:
        print("\nmodel_predicted_quality:")
        print(
            df["model_predicted_quality"]
            .value_counts(dropna=False)
            .to_string()
        )

print("\n========================================")
print("AUDIT COMPLETE")
print("========================================")
import pandas as pd
import joblib


# --------------------------------------------------
# File paths
# --------------------------------------------------

DATA_PATH = "data/actual_pcl_data.csv"
MODEL_PATH = "model/pcl_quality_model.pkl"
FEATURE_PATH = "model/feature_columns.pkl"


# --------------------------------------------------
# Load model, feature list and actual data
# --------------------------------------------------

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)

df = pd.read_csv(DATA_PATH)


# --------------------------------------------------
# Check that required input columns exist
# --------------------------------------------------

missing_features = [
    feature for feature in feature_columns
    if feature not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing required input features: {missing_features}"
    )


# --------------------------------------------------
# Handle empty actual dataset
# --------------------------------------------------

if df.empty:
    print("actual_pcl_data.csv contains no samples.")
    print("Add a new sample with the 26 input features first.")
    raise SystemExit(0)


# --------------------------------------------------
# Predict each sample
# --------------------------------------------------

X = df[feature_columns]

predictions = model.predict(X)
probabilities = model.predict_proba(X)

classes = list(model.classes_)

prediction_confidence = probabilities.max(axis=1)


# --------------------------------------------------
# Update prediction columns
# --------------------------------------------------

df["predicted_quality"] = predictions
df["prediction_confidence"] = prediction_confidence


# --------------------------------------------------
# Save updated dataset
# --------------------------------------------------

df.to_csv(DATA_PATH, index=False)


# --------------------------------------------------
# Display results
# --------------------------------------------------

print("========================================")
print("PCL QUALITY PREDICTION COMPLETE")
print("========================================")

print(f"Samples processed : {len(df)}")

for index, row in df.iterrows():
    print(
        f"{row['sample_id']} -> "
        f"{row['predicted_quality']} "
        f"(confidence: {row['prediction_confidence']:.4f})"
    )

print()
print(f"Updated dataset: {DATA_PATH}")
print("========================================")
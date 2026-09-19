import pandas as pd
import joblib

MASTER_FILE = "data/real_pcl_experimental_master_updated.csv"
MODEL_FILE = "model/pcl_mechanical_model.pkl"
FEATURE_FILE = "model/mechanical_feature_columns.pkl"
MECHANICAL_FILE = "data/real_mechanical_features.csv"

OUTPUT_FILE = "data/real_pcl_experimental_master_final.csv"

# Load files
master = pd.read_csv(MASTER_FILE)
mechanical = pd.read_csv(MECHANICAL_FILE)

model = joblib.load(MODEL_FILE)
features = joblib.load(FEATURE_FILE)

# Predict for each real experimental record
X = mechanical[features]

predictions = model.predict(X)
probabilities = model.predict_proba(X)

classes = list(model.classes_)

prediction_rows = []

for i, prediction in enumerate(predictions):
    predicted_index = classes.index(prediction)
    confidence = probabilities[i][predicted_index]

    prediction_rows.append({
        "record_id": mechanical.iloc[i]["record_id"],
        "sample_id": mechanical.iloc[i]["sample_id"],
        "model_predicted_quality": prediction,
        "model_prediction_confidence": confidence,
        "prediction_source": "MECHANICAL_MODEL_TRAINED_ON_SYNTHETIC_DATA"
    })

prediction_df = pd.DataFrame(prediction_rows)

# Remove these columns if they already exist so the merge remains clean
columns_to_remove = [
    "model_predicted_quality",
    "model_prediction_confidence",
    "prediction_source"
]

master = master.drop(
    columns=[c for c in columns_to_remove if c in master.columns]
)

# Merge prediction into master
final = master.merge(
    prediction_df,
    on=["record_id", "sample_id"],
    how="left"
)

# Save new final version
final.to_csv(OUTPUT_FILE, index=False)

print("\nREAL EXPERIMENTAL MASTER + MODEL PREDICTION")
print("============================================")

print("\nPrediction:")
print(prediction_df.to_string(index=False))

print("\nExperimental quality fields:")
print(final[[
    "record_id",
    "sample_id",
    "quality_class",
    "quality_score"
]].to_string(index=False))

print("\nAI prediction fields:")
print(final[[
    "record_id",
    "sample_id",
    "model_predicted_quality",
    "model_prediction_confidence",
    "prediction_source"
]].to_string(index=False))

print("\nSaved to:", OUTPUT_FILE)
print("Shape:", final.shape)
print("\nColumns:")
print(list(final.columns))
import pandas as pd
import joblib

MODEL_FILE = "model/pcl_mechanical_model.pkl"
FEATURE_FILE = "model/mechanical_feature_columns.pkl"
DATA_FILE = "data/real_mechanical_features.csv"

# Load model and features
model = joblib.load(MODEL_FILE)
features = joblib.load(FEATURE_FILE)

# Load real derived mechanical data
df = pd.read_csv(DATA_FILE)

# Select T1
t1 = df[df["sample_id"] == "T1"].copy()

if t1.empty:
    raise ValueError("Sample T1 was not found.")

# Prepare model input
X = t1[features]

# Prediction
prediction = model.predict(X)[0]
probabilities = model.predict_proba(X)[0]

classes = list(model.classes_)

# Probability of predicted class
predicted_index = classes.index(prediction)
confidence = probabilities[predicted_index]

print("\nREAL T1 MODEL PREDICTION")
print("========================")
print("Sample ID :", t1.iloc[0]["sample_id"])
print("Record ID :", t1.iloc[0]["record_id"])

print("\nMechanical features used:")
for feature in features:
    print(f"{feature}: {t1.iloc[0][feature]}")

print("\nPrediction")
print("----------")
print("Predicted quality :", prediction)
print(f"Confidence        : {confidence:.4f} ({confidence * 100:.2f}%)")

print("\nClass probabilities:")
for cls, probability in zip(classes, probabilities):
    print(f"{cls}: {probability:.4f} ({probability * 100:.2f}%)")

print("\nIMPORTANT:")
print(
    "This is a machine-learning prediction based on recovered "
    "experimental tensile data."
)
print(
    "It is NOT an experimentally confirmed GOOD/POOR label."
)
print(
    "The mechanical-only model was trained and tested on "
    "synthetic research-based data."
)
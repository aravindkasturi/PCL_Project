import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

TRAIN_FILE = "data/PCL_train.csv"
TEST_FILE = "data/PCL_test.csv"

MODEL_FILE = "model/pcl_mechanical_model.pkl"
FEATURE_FILE = "model/mechanical_feature_columns.pkl"
RESULT_FILE = "results/mechanical_model_evaluation.txt"

# Mechanical features available from the real tensile experiment
FEATURES = [
    "mech_max_force_n",
    "mech_displacement_at_max_mm",
    "mech_force_mean_n",
    "mech_force_std_n",
    "mech_curve_slope_n_per_mm",
    "mech_energy_proxy"
]

TARGET = "quality_class"

# Load datasets
train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

# Verify required columns
for col in FEATURES + [TARGET]:
    if col not in train.columns:
        raise ValueError(f"Missing column in training data: {col}")

    if col not in test.columns:
        raise ValueError(f"Missing column in test data: {col}")

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

# Train mechanical-only model
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# Find probability of GOOD
classes = list(model.classes_)
good_index = classes.index("GOOD")
good_probability = y_prob[:, good_index]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label="GOOD")
recall = recall_score(y_test, y_pred, pos_label="GOOD")
f1 = f1_score(y_test, y_pred, pos_label="GOOD")
roc_auc = roc_auc_score(
    (y_test == "GOOD").astype(int),
    good_probability
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["GOOD", "POOR"]
)

report = classification_report(
    y_test,
    y_pred,
    labels=["GOOD", "POOR"]
)

# Save model and feature list
joblib.dump(model, MODEL_FILE)
joblib.dump(FEATURES, FEATURE_FILE)

# Save evaluation
with open(RESULT_FILE, "w") as f:
    f.write("PCL MECHANICAL-ONLY MODEL - TEST EVALUATION\n")
    f.write("============================================\n\n")

    f.write("IMPORTANT:\n")
    f.write(
        "This model was trained and evaluated using the existing "
        "synthetic research-based PCL train/test datasets.\n"
    )
    f.write(
        "It has NOT been validated against experimentally labelled "
        "real PCL samples.\n\n"
    )

    f.write(f"Training samples : {len(train)}\n")
    f.write(f"Test samples     : {len(test)}\n")
    f.write(f"Features used     : {len(FEATURES)}\n\n")

    f.write("Features:\n")
    for feature in FEATURES:
        f.write(f"- {feature}\n")

    f.write("\nMetrics\n")
    f.write("-------\n")
    f.write(f"Accuracy  : {accuracy:.4f}\n")
    f.write(f"Precision : {precision:.4f}\n")
    f.write(f"Recall    : {recall:.4f}\n")
    f.write(f"F1-score  : {f1:.4f}\n")
    f.write(f"ROC-AUC   : {roc_auc:.4f}\n\n")

    f.write("Confusion Matrix\n")
    f.write("                 Predicted\n")
    f.write("              GOOD      POOR\n")
    f.write(f"Actual GOOD    {cm[0,0]:3d}       {cm[0,1]:3d}\n")
    f.write(f"Actual POOR    {cm[1,0]:3d}       {cm[1,1]:3d}\n\n")

    f.write("Classification Report\n")
    f.write("---------------------\n")
    f.write(report)

print("\nPCL MECHANICAL-ONLY MODEL TRAINING COMPLETE")
print("---------------------------------------------")
print("Training samples :", len(train))
print("Test samples     :", len(test))
print("Features used    :", len(FEATURES))
print("Target column    :", TARGET)

print("\nFeatures:")
for i, feature in enumerate(FEATURES, 1):
    print(f"{i}. {feature}")

print("\nTEST RESULTS")
print("------------")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nConfusion Matrix")
print(cm)

print("\nModel saved to:", MODEL_FILE)
print("Features saved to:", FEATURE_FILE)
print("Results saved to:", RESULT_FILE)
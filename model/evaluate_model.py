import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)


TEST_PATH = "data/PCL_test.csv"
MODEL_PATH = "model/pcl_quality_model.pkl"
FEATURE_PATH = "model/feature_columns.pkl"
RESULT_PATH = "results/evaluation_results.txt"


# --------------------------------------------------
# 1. Load test data
# --------------------------------------------------

test_df = pd.read_csv(TEST_PATH)

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)


# --------------------------------------------------
# 2. Prepare test features and target
# --------------------------------------------------

X_test = test_df[feature_columns]
y_test = test_df["quality_class"]


# --------------------------------------------------
# 3. Generate predictions
# --------------------------------------------------

y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)

classes = list(model.classes_)
good_index = classes.index("GOOD")

y_good_probability = y_probability[:, good_index]


# --------------------------------------------------
# 4. Calculate metrics
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    pos_label="GOOD",
)

recall = recall_score(
    y_test,
    y_pred,
    pos_label="GOOD",
)

f1 = f1_score(
    y_test,
    y_pred,
    pos_label="GOOD",
)

roc_auc = roc_auc_score(
    (y_test == "GOOD").astype(int),
    y_good_probability,
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["GOOD", "POOR"],
)

report = classification_report(
    y_test,
    y_pred,
    labels=["GOOD", "POOR"],
)


# --------------------------------------------------
# 5. Display results
# --------------------------------------------------

print("========================================")
print("PCL QUALITY MODEL - TEST EVALUATION")
print("========================================")

print(f"Test samples : {len(test_df)}")
print(f"Features used: {len(feature_columns)}")
print()

print(f"Accuracy     : {accuracy:.4f}")
print(f"Precision    : {precision:.4f}")
print(f"Recall       : {recall:.4f}")
print(f"F1-score     : {f1:.4f}")
print(f"ROC-AUC      : {roc_auc:.4f}")

print()
print("Confusion Matrix")
print("                 Predicted")
print("              GOOD      POOR")
print(f"Actual GOOD   {cm[0, 0]:4d}      {cm[0, 1]:4d}")
print(f"Actual POOR   {cm[1, 0]:4d}      {cm[1, 1]:4d}")

print()
print("Classification Report")
print(report)


# --------------------------------------------------
# 6. Save evaluation results
# --------------------------------------------------

with open(RESULT_PATH, "w", encoding="utf-8") as file:

    file.write("PCL QUALITY MODEL - TEST EVALUATION\n")
    file.write("=" * 45 + "\n\n")

    file.write(f"Test samples : {len(test_df)}\n")
    file.write(f"Features used: {len(feature_columns)}\n\n")

    file.write(f"Accuracy     : {accuracy:.4f}\n")
    file.write(f"Precision    : {precision:.4f}\n")
    file.write(f"Recall       : {recall:.4f}\n")
    file.write(f"F1-score     : {f1:.4f}\n")
    file.write(f"ROC-AUC      : {roc_auc:.4f}\n\n")

    file.write("Confusion Matrix\n")
    file.write("                 Predicted\n")
    file.write("              GOOD      POOR\n")
    file.write(f"Actual GOOD   {cm[0, 0]:4d}      {cm[0, 1]:4d}\n")
    file.write(f"Actual POOR   {cm[1, 0]:4d}      {cm[1, 1]:4d}\n\n")

    file.write("Classification Report\n")
    file.write(report)

print()
print(f"Results saved to: {RESULT_PATH}")
print("========================================")
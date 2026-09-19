import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier


# --------------------------------------------------
# 1. Load training data
# --------------------------------------------------

TRAIN_PATH = "data/PCL_train.csv"
MODEL_PATH = "model/pcl_quality_model.pkl"
FEATURE_PATH = "model/feature_columns.pkl"

train_df = pd.read_csv(TRAIN_PATH)


# --------------------------------------------------
# 2. Define predictor features
# --------------------------------------------------

feature_columns = [
    "img_mean_intensity",
    "img_intensity_std",
    "img_texture_contrast",
    "img_texture_energy",
    "img_edge_density",
    "img_fiber_density",
    "img_orientation_index",
    "img_porosity_est",
    "img_mean_fiber_width_px",
    "img_fiber_width_std_px",
    "spec_peak1_nm",
    "spec_peak1_intensity",
    "spec_peak2_nm",
    "spec_peak2_intensity",
    "spec_peak3_nm",
    "spec_peak3_intensity",
    "spec_area_norm",
    "spec_mean",
    "spec_std",
    "spec_baseline_slope",
    "mech_max_force_n",
    "mech_displacement_at_max_mm",
    "mech_force_mean_n",
    "mech_force_std_n",
    "mech_curve_slope_n_per_mm",
    "mech_energy_proxy",
]


# --------------------------------------------------
# 3. Separate features and target
# --------------------------------------------------

X_train = train_df[feature_columns]
y_train = train_df["quality_class"]


# --------------------------------------------------
# 4. Create baseline model
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)


# --------------------------------------------------
# 5. Train model
# --------------------------------------------------

model.fit(X_train, y_train)


# --------------------------------------------------
# 6. Save model and feature structure
# --------------------------------------------------

joblib.dump(model, MODEL_PATH)
joblib.dump(feature_columns, FEATURE_PATH)


# --------------------------------------------------
# 7. Training summary
# --------------------------------------------------

print("========================================")
print("PCL QUALITY MODEL TRAINING COMPLETE")
print("========================================")
print(f"Training samples : {len(train_df)}")
print(f"Features used    : {len(feature_columns)}")
print(f"Target column    : quality_class")
print(f"Classes          : {sorted(y_train.unique())}")
print()
print(f"Model saved to   : {MODEL_PATH}")
print(f"Features saved to: {FEATURE_PATH}")
print("========================================")
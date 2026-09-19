import os

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# Paths
# --------------------------------------------------

TRAIN_PATH = "data/PCL_train.csv"
DATA_PATH = "data/actual_pcl_data.csv"
MODEL_PATH = "model/pcl_quality_model.pkl"
FEATURE_PATH = "model/feature_columns.pkl"
OUTPUT_DIR = "results"


# --------------------------------------------------
# Load data and model
# --------------------------------------------------

train_df = pd.read_csv(TRAIN_PATH)
df = pd.read_csv(DATA_PATH)

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)


# --------------------------------------------------
# Check actual data
# --------------------------------------------------

if df.empty:
    raise ValueError(
        "No samples found in actual_pcl_data.csv"
    )


# --------------------------------------------------
# Create output directory
# --------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# Calculate training-data ranges
#
# Used only for visualization.
# The model itself is NOT changed.
# --------------------------------------------------

train_min = train_df[feature_columns].min()
train_max = train_df[feature_columns].max()

train_range = train_max - train_min

train_range = train_range.replace(0, 1)

# --------------------------------------------------
# Generate visualization for every sample
# --------------------------------------------------

for _, sample in df.iterrows():

    sample_id = str(sample["sample_id"])

    # ----------------------------------------------
    # Sample values
    # ----------------------------------------------

    values = sample[
        feature_columns
    ].astype(float)

    # ----------------------------------------------
    # Normalize using TRAINING data ranges
    #
    # This is visualization only.
    # ----------------------------------------------

    normalized_values = (
        (values - train_min)
        / train_range
    )

    normalized_values = normalized_values.clip(
        0,
        1,
    )

    # ----------------------------------------------
    # Model feature importance
    # ----------------------------------------------

    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=True,
    )

    # ----------------------------------------------
    # Prediction probabilities
    # ----------------------------------------------

    X = pd.DataFrame(
        [values.values],
        columns=feature_columns,
    )

    probabilities = model.predict_proba(X)[0]

    classes = list(model.classes_)

    probability_dict = dict(
        zip(classes, probabilities)
    )

    predicted_quality = model.predict(X)[0]

    confidence = max(probabilities)

    # ----------------------------------------------
    # Create figure
    # ----------------------------------------------

    fig = plt.figure(
        figsize=(17, 9)
    )

    fig.suptitle(
        "PCL Non-Woven Fabric Quality Detection",
        fontsize=22,
        fontweight="bold",
        y=0.98,
    )

    # ==================================================
    # PANEL 1 — SAMPLE PROPERTY HEATMAP
    # ==================================================

    ax1 = plt.subplot(2, 2, 1)

    heatmap_values = normalized_values.values.reshape(
        -1,
        1,
    )

    image = ax1.imshow(
        heatmap_values,
        aspect="auto",
        interpolation="nearest",
    )

    ax1.set_title(
        "PCL Sample Property Heatmap",
        fontsize=15,
        fontweight="bold",
    )

    ax1.set_xlabel(
        "Selected Sample"
    )

    ax1.set_ylabel(
        "Measured Property"
    )

    ax1.set_xticks([0])
    ax1.set_xticklabels([sample_id])

    ax1.set_yticks(
        range(len(feature_columns))
    )

    ax1.set_yticklabels(
        feature_columns,
        fontsize=7,
    )

    fig.colorbar(
        image,
        ax=ax1,
        fraction=0.046,
        pad=0.04,
        label="Normalized Property Value",
    )

    # ==================================================
    # PANEL 2 — GLOBAL MODEL IMPORTANCE
    # ==================================================

    ax2 = plt.subplot(2, 2, 2)

    top_features = importance_df.tail(10)

    ax2.barh(
        top_features["feature"],
        top_features["importance"],
    )

    ax2.set_title(
        "Important PCL Features",
        fontsize=15,
        fontweight="bold",
    )

    ax2.set_xlabel(
        "Random Forest Feature Importance"
    )

    # ==================================================
    # PANEL 3 — QUALITY PROBABILITY
    # ==================================================

    ax3 = plt.subplot(2, 2, 3)

    good_probability = probability_dict.get(
        "GOOD",
        0,
    )

    poor_probability = probability_dict.get(
        "POOR",
        0,
    )

    probability_values = [
        good_probability,
        poor_probability,
    ]

    bars = ax3.bar(
        ["GOOD", "POOR"],
        probability_values,
    )

    ax3.set_ylim(
        0,
        1,
    )

    ax3.set_ylabel(
        "Prediction Probability"
    )

    ax3.set_title(
        "AI Quality Prediction",
        fontsize=15,
        fontweight="bold",
    )

    for bar, value in zip(
        bars,
        probability_values,
    ):
        ax3.text(
            bar.get_x()
            + bar.get_width() / 2,
            min(value + 0.03, 0.96),
            f"{value * 100:.1f}%",
            ha="center",
            fontweight="bold",
        )

    # ==================================================
    # PANEL 4 — SAMPLE PROPERTY PROFILE
    # ==================================================

    ax4 = plt.subplot(2, 2, 4)

    ax4.plot(
        range(1, len(feature_columns) + 1),
        normalized_values.values,
        marker="o",
    )

    ax4.set_title(
        "PCL Sample Property Profile",
        fontsize=15,
        fontweight="bold",
    )

    ax4.set_xlabel(
        "Measured Property Number"
    )

    ax4.set_ylabel(
        "Normalized Value"
    )

    ax4.set_ylim(
        0,
        1,
    )

    ax4.set_xticks(
        range(1, len(feature_columns) + 1)
    )

    ax4.tick_params(
        axis="x",
        labelsize=7,
    )

    ax4.grid(
        True,
        alpha=0.3,
    )

    # ----------------------------------------------
    # Bottom result information
    # ----------------------------------------------

    fig.text(
        0.5,
        0.015,
        (
            f"Sample ID: {sample_id}    |    "
            f"Predicted Quality: {predicted_quality}    |    "
            f"Confidence: {confidence * 100:.2f}%"
        ),
        ha="center",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout(
        rect=[0, 0.05, 1, 0.94]
    )

    # ----------------------------------------------
    # Save
    # ----------------------------------------------

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{sample_id}_analysis.png",
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Generated visualization: {output_path}"
    )


print()
print("========================================")
print("VISUALIZATION GENERATION COMPLETE")
print("========================================")
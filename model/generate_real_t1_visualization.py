import pandas as pd
import matplotlib.pyplot as plt

POINTS_FILE = "data/real_tensile_verified_points.csv"
MASTER_FILE = "data/real_pcl_experimental_master_final.csv"

OUTPUT_FILE = "results/EXP_T1_real_tensile_analysis.png"

# Load recovered tensile points
points = pd.read_csv(POINTS_FILE)

# Load final real experimental record
master = pd.read_csv(MASTER_FILE)

t1 = master[master["sample_id"] == "T1"]

if t1.empty:
    raise ValueError("T1 was not found in the final master dataset.")

t1 = t1.iloc[0]

# Maximum force point from recovered points
max_idx = points["force_n"].idxmax()
max_force = points.loc[max_idx, "force_n"]
max_disp = points.loc[max_idx, "displacement_mm"]

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    points["displacement_mm"],
    points["force_n"],
    marker="o",
    linewidth=2,
    markersize=4,
    label="Recovered tensile points"
)

# Mark maximum force
ax.scatter(
    max_disp,
    max_force,
    s=100,
    zorder=5,
    label=f"Maximum force = {max_force:.3f} N"
)

ax.annotate(
    f"Max force: {max_force:.3f} N\n"
    f"At displacement: {max_disp:.3f} mm",
    xy=(max_disp, max_force),
    xytext=(max_disp + 1, max_force - 0.15),
    arrowprops=dict(arrowstyle="->")
)

ax.set_title(
    "T1 PCL Non-Woven Fabric — Recovered Tensile Analysis"
)

ax.set_xlabel("Displacement (mm)")
ax.set_ylabel("Force (N)")

ax.grid(True, alpha=0.3)
ax.legend()

# Add experimental/model information
text = (
    f"Sample: T1\n"
    f"Test: {t1['test_name']}\n"
    f"Temperature: {t1['temperature_c']:.0f} °C\n"
    f"AI prediction: {t1['model_predicted_quality']}\n"
    f"Prediction confidence: "
    f"{t1['model_prediction_confidence'] * 100:.2f}%\n\n"
    "Data source: recovered readable points from\n"
    "experimental screenshots/video.\n"
    "Not a complete machine-export dataset."
)

ax.text(
    0.98,
    0.03,
    text,
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=9,
    bbox=dict(boxstyle="round", alpha=0.9)
)

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nREAL T1 VISUALIZATION COMPLETE")
print("--------------------------------")
print("Sample:", t1["sample_id"])
print("Maximum force:", max_force, "N")
print("Displacement at maximum force:", max_disp, "mm")
print("AI prediction:", t1["model_predicted_quality"])
print(
    "Prediction confidence:",
    f"{t1['model_prediction_confidence'] * 100:.2f}%"
)
print("\nSaved to:", OUTPUT_FILE)
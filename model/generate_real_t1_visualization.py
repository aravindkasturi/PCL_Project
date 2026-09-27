import pandas as pd
import matplotlib.pyplot as plt

POINTS_FILE = "data/real_tensile_verified_points.csv"
MASTER_FILE = "data/real_pcl_experimental_master_final.csv"

OUTPUT_FILE = "results/EXP_T1_real_tensile_analysis.png"

# ---------------------------------------------------------
# Load recovered tensile points
# ---------------------------------------------------------
points = pd.read_csv(POINTS_FILE)

# Sort by displacement
points = points.sort_values("displacement_mm").reset_index(drop=True)

# ---------------------------------------------------------
# Load final real experimental record
# ---------------------------------------------------------
master = pd.read_csv(MASTER_FILE)

t1 = master[master["sample_id"] == "T1"]

if t1.empty:
    raise ValueError("T1 was not found in the final master dataset.")

t1 = t1.iloc[0]

# ---------------------------------------------------------
# Maximum force point
# ---------------------------------------------------------
max_idx = points["force_n"].idxmax()

max_force = points.loc[max_idx, "force_n"]
max_disp = points.loc[max_idx, "displacement_mm"]

# ---------------------------------------------------------
# Detect largest consecutive force drop
# ---------------------------------------------------------
points["force_drop_n"] = (
    points["force_n"].shift(1) - points["force_n"]
)

break_idx = points["force_drop_n"].idxmax()

before = points.loc[break_idx - 1]
after = points.loc[break_idx]

break_start_disp = before["displacement_mm"]
break_end_disp = after["displacement_mm"]

break_start_force = before["force_n"]
break_end_force = after["force_n"]

force_drop = after["force_drop_n"]

# ---------------------------------------------------------
# Create figure
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    points["displacement_mm"],
    points["force_n"],
    marker="o",
    linewidth=2,
    markersize=4,
    label="Recovered tensile points"
)

# ---------------------------------------------------------
# Highlight estimated failure / break region
# ---------------------------------------------------------
ax.axvspan(
    break_start_disp,
    break_end_disp,
    alpha=0.25,
    label="Estimated failure / break region"
)

# ---------------------------------------------------------
# Mark maximum force
# ---------------------------------------------------------
ax.scatter(
    max_disp,
    max_force,
    s=100,
    zorder=5,
    label=f"Maximum force = {max_force:.3f} N"
)

# Maximum-force annotation
ax.annotate(
    f"Maximum force: {max_force:.3f} N\n"
    f"Displacement: {max_disp:.3f} mm",
    xy=(max_disp, max_force),
    xytext=(10.8, 0.95),
    arrowprops=dict(
        arrowstyle="->",
        linewidth=1.2
    ),
    fontsize=9,
    bbox=dict(
        boxstyle="round",
        alpha=0.85
    )
)

# ---------------------------------------------------------
# Mark beginning and end of failure region
# ---------------------------------------------------------
ax.scatter(
    break_start_disp,
    break_start_force,
    s=80,
    zorder=6
)

ax.scatter(
    break_end_disp,
    break_end_force,
    s=80,
    zorder=6
)

# ---------------------------------------------------------
# Failure-region annotation
# ---------------------------------------------------------
ax.annotate(
    f"Estimated failure region\n"
    f"{break_start_disp:.3f}–{break_end_disp:.3f} mm\n"
    f"Force drop: {force_drop:.3f} N",
    xy=(
        (break_start_disp + break_end_disp) / 2,
        (break_start_force + break_end_force) / 2
    ),
    xytext=(16.5, 0.48),
    arrowprops=dict(
        arrowstyle="->",
        linewidth=1.2
    ),
    fontsize=9,
    bbox=dict(
        boxstyle="round",
        alpha=0.85
    )
)

# ---------------------------------------------------------
# Axis labels and title
# ---------------------------------------------------------
ax.set_title(
    "T1 PCL Non-Woven Fabric — Recovered Tensile Analysis"
)

ax.set_xlabel("Displacement (mm)")
ax.set_ylabel("Force (N)")

ax.grid(True, alpha=0.3)

ax.legend(
    loc="upper left"
)

# ---------------------------------------------------------
# Information box
# ---------------------------------------------------------
text = (
    f"Sample: T1\n"
    f"Test: {t1['test_name']}\n"
    f"Temperature: {t1['temperature_c']:.0f} °C\n"
    f"Maximum force: {max_force:.3f} N\n"
    f"Estimated failure region: "
    f"{break_start_disp:.3f}–{break_end_disp:.3f} mm\n"
    f"AI prediction: {t1['model_predicted_quality']}\n"
    f"Prediction confidence: "
    f"{t1['model_prediction_confidence'] * 100:.2f}%\n\n"
    "Data source: recovered readable points from\n"
    "experimental screenshots/video.\n"
    "Not a complete machine-export dataset."
)

ax.text(
    0.02,
    0.03,
    text,
    transform=ax.transAxes,
    ha="left",
    va="bottom",
    fontsize=9,
    bbox=dict(
        boxstyle="round",
        alpha=0.9
    )
)

# ---------------------------------------------------------
# Save figure
# ---------------------------------------------------------
plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ---------------------------------------------------------
# Console output
# ---------------------------------------------------------
print("\nREAL T1 VISUALIZATION COMPLETE")
print("--------------------------------")
print("Sample:", t1["sample_id"])
print("Maximum force:", max_force, "N")
print("Displacement at maximum force:", max_disp, "mm")

print(
    "Estimated failure region:",
    f"{break_start_disp:.3f} - {break_end_disp:.3f} mm"
)

print(
    "Force drop:",
    f"{force_drop:.3f}",
    "N"
)

print(
    "AI prediction:",
    t1["model_predicted_quality"]
)

print(
    "Prediction confidence:",
    f"{t1['model_prediction_confidence'] * 100:.2f}%"
)

print("\nSaved to:", OUTPUT_FILE)
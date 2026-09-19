import pandas as pd
import numpy as np

INPUT_FILE = "data/real_tensile_verified_points.csv"
OUTPUT_FILE = "data/real_mechanical_features.csv"

# Load recovered tensile points
df = pd.read_csv(INPUT_FILE)

# Convert numeric columns
df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
df["displacement_mm"] = pd.to_numeric(df["displacement_mm"], errors="coerce")
df["force_n"] = pd.to_numeric(df["force_n"], errors="coerce")

# Remove invalid rows
df = df.dropna(subset=["time_s", "displacement_mm", "force_n"])

# Sort by displacement
df = df.sort_values("displacement_mm").reset_index(drop=True)

# Basic mechanical features
max_force_idx = df["force_n"].idxmax()

max_force = df.loc[max_force_idx, "force_n"]
displacement_at_max = df.loc[max_force_idx, "displacement_mm"]

force_mean = df["force_n"].mean()
force_std = df["force_n"].std()

# Linear force-displacement slope
slope = np.polyfit(
    df["displacement_mm"],
    df["force_n"],
    1
)[0]

# Trapezoidal area under force-displacement curve
energy_proxy = np.trapezoid(
    df["force_n"],
    df["displacement_mm"]
)

# Create output
result = pd.DataFrame([{
    "record_id": df.loc[0, "record_id"],
    "sample_id": df.loc[0, "sample_id"],
    "mech_max_force_n": max_force,
    "mech_displacement_at_max_mm": displacement_at_max,
    "mech_force_mean_n": force_mean,
    "mech_force_std_n": force_std,
    "mech_curve_slope_n_per_mm": slope,
    "mech_energy_proxy": energy_proxy,
    "source": "REAL_RECOVERED_FROM_SCREENSHOTS",
    "data_status": "DERIVED_FROM_RECOVERED_POINTS"
}])

result.to_csv(OUTPUT_FILE, index=False)

print("\nREAL MECHANICAL FEATURE EXTRACTION COMPLETE")
print("--------------------------------------------")
print(result.to_string(index=False))
print("\nSaved to:", OUTPUT_FILE)
print("\nShape:", result.shape)
print("\nColumns:", list(result.columns))
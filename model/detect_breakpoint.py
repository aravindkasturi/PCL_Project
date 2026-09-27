import pandas as pd

DATA_PATH = "data/real_tensile_verified_points.csv"

# Load verified tensile points
df = pd.read_csv(DATA_PATH)

# Sort by displacement
df = df.sort_values("displacement_mm").reset_index(drop=True)

# Calculate force drop between consecutive points
df["force_drop_n"] = df["force_n"].shift(1) - df["force_n"]

# Find the largest force drop
break_index = df["force_drop_n"].idxmax()

# Point before the major drop
before = df.loc[break_index - 1]

# Point after the major drop
after = df.loc[break_index]

print("\n" + "=" * 55)
print("PCL T1 BREAKPOINT / FAILURE REGION DETECTION")
print("=" * 55)

print(f"\nPoint before major force drop:")
print(f"Displacement : {before['displacement_mm']:.3f} mm")
print(f"Force        : {before['force_n']:.3f} N")

print(f"\nPoint after major force drop:")
print(f"Displacement : {after['displacement_mm']:.3f} mm")
print(f"Force        : {after['force_n']:.3f} N")

print(f"\nForce drop   : {after['force_drop_n']:.3f} N")

print(
    f"\nEstimated failure/break region:"
    f" {before['displacement_mm']:.3f} - "
    f"{after['displacement_mm']:.3f} mm"
)

print("\nNote:")
print(
    "This is an estimated failure region based on the largest "
    "consecutive force drop in the recovered tensile points."
)
print(
    "It is not an exact physical break timestamp because the "
    "complete raw tensile-test export is not available."
)

print("=" * 55)
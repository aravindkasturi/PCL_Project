import pandas as pd


def analyze_tensile_curve(df):
    """
    Analyze a tensile force-displacement curve.

    Required columns:
        displacement_mm
        force_n

    Returns:
        Dictionary containing maximum force and
        estimated failure/break region.
    """

    required_columns = [
        "displacement_mm",
        "force_n"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Keep only required numeric data
    points = df[
        ["displacement_mm", "force_n"]
    ].copy()

    points["displacement_mm"] = pd.to_numeric(
        points["displacement_mm"],
        errors="coerce"
    )

    points["force_n"] = pd.to_numeric(
        points["force_n"],
        errors="coerce"
    )

    points = points.dropna()

    if len(points) < 2:
        raise ValueError(
            "At least two valid tensile points are required."
        )

    # Sort by displacement
    points = (
        points
        .sort_values("displacement_mm")
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # Maximum force
    # ---------------------------------------------------------

    max_idx = points["force_n"].idxmax()

    max_force = float(
        points.loc[max_idx, "force_n"]
    )

    max_force_displacement = float(
        points.loc[max_idx, "displacement_mm"]
    )

    # ---------------------------------------------------------
    # Largest consecutive force drop
    # ---------------------------------------------------------

    points["force_drop_n"] = (
        points["force_n"].shift(1)
        - points["force_n"]
    )

    break_idx = points["force_drop_n"].idxmax()

    if (
        pd.isna(break_idx)
        or break_idx <= 0
    ):
        raise ValueError(
            "Unable to identify a failure region."
        )

    before = points.loc[break_idx - 1]
    after = points.loc[break_idx]

    break_start_displacement = float(
        before["displacement_mm"]
    )

    break_end_displacement = float(
        after["displacement_mm"]
    )

    break_start_force = float(
        before["force_n"]
    )

    break_end_force = float(
        after["force_n"]
    )

    force_drop = float(
        after["force_drop_n"]
    )

    return {
        "max_force_n": max_force,
        "displacement_at_max_force_mm":
            max_force_displacement,
        "failure_start_mm":
            break_start_displacement,
        "failure_end_mm":
            break_end_displacement,
        "failure_start_force_n":
            break_start_force,
        "failure_end_force_n":
            break_end_force,
        "force_drop_n":
            force_drop
    }


# ---------------------------------------------------------
# Test using the recovered T1 data
# ---------------------------------------------------------

if __name__ == "__main__":

    DATA_PATH = (
        "data/real_tensile_verified_points.csv"
    )

    df = pd.read_csv(DATA_PATH)

    result = analyze_tensile_curve(df)

    print("\n" + "=" * 55)
    print("TENSILE CURVE ANALYSIS")
    print("=" * 55)

    print(
        "\nMaximum force:",
        f"{result['max_force_n']:.3f} N"
    )

    print(
        "Displacement at maximum force:",
        f"{result['displacement_at_max_force_mm']:.3f} mm"
    )

    print(
        "\nEstimated failure region:",
        f"{result['failure_start_mm']:.3f}"
        f" - "
        f"{result['failure_end_mm']:.3f} mm"
    )

    print(
        "Force drop:",
        f"{result['force_drop_n']:.3f} N"
    )

    print("\nNote:")
    print(
        "Failure region is estimated from the "
        "largest consecutive force drop."
    )

    print("=" * 55)
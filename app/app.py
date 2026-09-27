import os
import sys
import joblib
import re
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Load the tensile analyzer directly from the model folder.
# This avoids relying on Python package-path behavior in Streamlit Cloud.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ANALYZER_PATH = os.path.join(PROJECT_ROOT, "model", "analyze_tensile_curve.py")

import importlib.util

spec = importlib.util.spec_from_file_location("analyze_tensile_curve_module", ANALYZER_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load tensile analyzer from: {ANALYZER_PATH}")
analyzer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyzer_module)
analyze_tensile_curve = analyzer_module.analyze_tensile_curve


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PCL Quality Detection",
    page_icon="🧪",
    layout="wide",
)


# ============================================================
# PROJECT PATHS
# ============================================================

DATA_PATH = "data/actual_pcl_data.csv"

MODEL_PATH = "model/pcl_quality_model.pkl"
FEATURE_PATH = "model/feature_columns.pkl"

TRAIN_PATH = "data/PCL_train.csv"

REAL_MASTER_PATH = (
    "data/real_pcl_experimental_master_final.csv"
)

RESULTS_DIR = "results"

REAL_POINTS_PATH = "data/real_tensile_verified_points.csv"
TENSILE_CURVES_PATH = "data/tensile_curve_data.csv"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

# Store tensile analysis results for newly added samples.
if "tensile_analysis_results" not in st.session_state:
    st.session_state["tensile_analysis_results"] = {}

if "tensile_curve_data" not in st.session_state:
    st.session_state["tensile_curve_data"] = {}


# Load previously saved tensile curves so breakpoint analysis survives
# Streamlit restarts/redeployments.
if os.path.exists(TENSILE_CURVES_PATH):
    try:
        saved_tensile = pd.read_csv(TENSILE_CURVES_PATH)
        if "sample_id" in saved_tensile.columns:
            for saved_id, saved_group in saved_tensile.groupby("sample_id"):
                curve_df = saved_group.drop(columns=["sample_id"]).copy()
                st.session_state["tensile_curve_data"][str(saved_id)] = curve_df
                st.session_state["tensile_analysis_results"][str(saved_id)] = analyze_tensile_curve(curve_df)
    except Exception:
        pass


# ============================================================
# LOAD MAIN AI MODEL
# ============================================================

try:

    df = pd.read_csv(DATA_PATH)

    model = joblib.load(
        MODEL_PATH
    )

    feature_columns = joblib.load(
        FEATURE_PATH
    )

except Exception as e:

    st.error(
        "Unable to load the PCL AI project files."
    )

    st.exception(e)

    st.stop()


# ============================================================
# FEATURE GROUPS
# ============================================================

image_features = [
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
]


spectroscopy_features = [
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
]


mechanical_features = [
    "mech_max_force_n",
    "mech_displacement_at_max_mm",
    "mech_force_mean_n",
    "mech_force_std_n",
    "mech_curve_slope_n_per_mm",
    "mech_energy_proxy",
]


# ============================================================
# DISPLAY NAMES
# ============================================================

display_names = {

    "img_mean_intensity":
        "Mean Intensity",

    "img_intensity_std":
        "Intensity Std",

    "img_texture_contrast":
        "Texture Contrast",

    "img_texture_energy":
        "Texture Energy",

    "img_edge_density":
        "Edge Density",

    "img_fiber_density":
        "Fiber Density",

    "img_orientation_index":
        "Orientation Index",

    "img_porosity_est":
        "Porosity Estimate",

    "img_mean_fiber_width_px":
        "Mean Fiber Width (px)",

    "img_fiber_width_std_px":
        "Fiber Width Std (px)",

    "spec_peak1_nm":
        "Peak 1 Wavelength (nm)",

    "spec_peak1_intensity":
        "Peak 1 Intensity",

    "spec_peak2_nm":
        "Peak 2 Wavelength (nm)",

    "spec_peak2_intensity":
        "Peak 2 Intensity",

    "spec_peak3_nm":
        "Peak 3 Wavelength (nm)",

    "spec_peak3_intensity":
        "Peak 3 Intensity",

    "spec_area_norm":
        "Area Normalized",

    "spec_mean":
        "Spectroscopy Mean",

    "spec_std":
        "Spectroscopy Std",

    "spec_baseline_slope":
        "Baseline Slope",

    "mech_max_force_n":
        "Maximum Force (N)",

    "mech_displacement_at_max_mm":
        "Displacement at Maximum Force (mm)",

    "mech_force_mean_n":
        "Mean Force (N)",

    "mech_force_std_n":
        "Force Std (N)",

    "mech_curve_slope_n_per_mm":
        "Curve Slope (N/mm)",

    "mech_energy_proxy":
        "Energy Proxy",
}


# ============================================================
# HEADER
# ============================================================

st.title(
    "🧪 PCL Non-Woven Fabric Quality Detection"
)

st.caption(
    "AI-assisted quality assessment using image, "
    "spectroscopy, and mechanical properties"
)

st.divider()


# ============================================================
# MAIN APPLICATION TABS
# ============================================================

real_tab, ai_tab = st.tabs(
    [
        "🔬 Real Experimental Analysis",
        "🤖 AI Quality Prediction",
    ]
)


# ============================================================
# ============================================================
# REAL EXPERIMENTAL ANALYSIS
# ============================================================
# ============================================================

with real_tab:

    st.header(
        "🔬 Real Experimental Samples"
    )

    st.caption(
        "Experimental PCL data recovered from available "
        "laboratory screenshots, video, and sample records."
    )

    if not os.path.exists(
        REAL_MASTER_PATH
    ):

        st.warning(
            "Real experimental master dataset was not found."
        )

    else:

        try:

            real_df = pd.read_csv(
                REAL_MASTER_PATH
            )

            if real_df.empty:

                st.info(
                    "No real experimental samples are currently available."
                )

            else:

                # ------------------------------------------------
                # SAMPLE SELECTION
                # ------------------------------------------------

                real_sample_ids = (
                    real_df["sample_id"]
                    .dropna()
                    .astype(str)
                    .tolist()
                )

                selected_real_sample = st.selectbox(
                    "Select Experimental Sample",
                    real_sample_ids,
                    key="real_experimental_sample",
                )

                real_sample = real_df[
                    real_df["sample_id"].astype(str)
                    == selected_real_sample
                ].iloc[0]


                # ------------------------------------------------
                # SAMPLE HEADER
                # ------------------------------------------------

                st.subheader(
                    f"Experimental Sample: {selected_real_sample}"
                )


                c1, c2, c3, c4 = st.columns(4)


                with c1:

                    st.metric(
                        "Sample ID",
                        selected_real_sample
                    )


                with c2:

                    st.metric(
                        "Test",
                        str(
                            real_sample[
                                "test_name"
                            ]
                        )
                    )


                with c3:

                    temperature = float(
                        real_sample[
                            "temperature_c"
                        ]
                    )

                    st.metric(
                        "Temperature",
                        f"{temperature:.0f} °C"
                    )


                with c4:

                    max_force = float(
                        real_sample[
                            "max_force_n"
                        ]
                    )

                    st.metric(
                        "Maximum Force",
                        f"{max_force:.3f} N"
                    )


                st.divider()


                # ------------------------------------------------
                # EXPERIMENTAL STATUS
                # ------------------------------------------------

                st.subheader(
                    "Experimental Status"
                )


                c1, c2, c3 = st.columns(3)


                with c1:

                    experimental_quality = (
                        real_sample[
                            "quality_class"
                        ]
                    )

                    if pd.isna(
                        experimental_quality
                    ):

                        st.metric(
                            "Experimental Quality",
                            "Not available"
                        )

                    else:

                        st.metric(
                            "Experimental Quality",
                            str(
                                experimental_quality
                            )
                        )


                with c2:

                    predicted_quality = (
                        real_sample[
                            "model_predicted_quality"
                        ]
                    )

                    st.metric(
                        "AI Prediction",
                        str(
                            predicted_quality
                        )
                    )


                with c3:

                    confidence = float(
                        real_sample[
                            "model_prediction_confidence"
                        ]
                    )

                    st.metric(
                        "AI Confidence",
                        f"{confidence * 100:.2f}%"
                    )


                st.info(
                    "The AI prediction is not experimental "
                    "ground truth. The mechanical-only model "
                    "was trained and tested on synthetic "
                    "research-based data."
                )


                # ------------------------------------------------
                # RECOVERED MECHANICAL FEATURES
                # ------------------------------------------------

                st.subheader(
                    "Recovered Mechanical Properties"
                )


                for i in range(
                    0,
                    len(mechanical_features),
                    2
                ):

                    cols = st.columns(2)


                    for j, feature in enumerate(
                        mechanical_features[
                            i:i + 2
                        ]
                    ):

                        value = (
                            real_sample[
                                feature
                            ]
                        )


                        if pd.isna(value):

                            value_text = (
                                "Not available"
                            )

                        else:

                            value_text = (
                                f"{float(value):.6f}"
                            )


                        with cols[j]:

                            st.metric(
                                display_names.get(
                                    feature,
                                    feature
                                ),
                                value_text
                            )


                # ------------------------------------------------
                # RECOVERED EXPERIMENTAL MEASUREMENTS
                # ------------------------------------------------

                st.subheader(
                    "Recovered Experimental Measurements"
                )


                measurement_data = {

                    "Measurement": [

                        "Initial Size",

                        "Maximum Recorded Displacement",

                        "Maximum Recorded Size",

                        "Final Recorded Time",

                        "Initial Force",

                    ],

                    "Value": [

                        f"{float(real_sample['initial_size_mm']):.3f} mm",

                        f"{float(real_sample['max_recorded_displacement_mm']):.3f} mm",

                        f"{float(real_sample['max_recorded_size_mm']):.3f} mm",

                        f"{float(real_sample['final_recorded_time_s']):.3f} s",

                        f"{float(real_sample['initial_force_n']):.3f} N",

                    ]
                }


                st.dataframe(
                    pd.DataFrame(
                        measurement_data
                    ),
                    use_container_width=True,
                    hide_index=True,
                )


                # ------------------------------------------------
                # ESTIMATED FAILURE / BREAK REGION
                # ------------------------------------------------

                st.subheader(
                    "Estimated Failure / Break Region"
                )

                if os.path.exists(REAL_POINTS_PATH):

                    try:

                        tensile_points = pd.read_csv(
                            REAL_POINTS_PATH
                        )

                        # Use the selected experimental sample only.
                        tensile_points = tensile_points[
                            tensile_points["sample_id"].astype(str)
                            == selected_real_sample
                        ].copy()

                        if len(tensile_points) >= 2:

                            tensile_points = (
                                tensile_points
                                .sort_values("displacement_mm")
                                .reset_index(drop=True)
                            )

                            # Calculate consecutive force drops.
                            tensile_points["force_drop_n"] = (
                                tensile_points["force_n"].shift(1)
                                - tensile_points["force_n"]
                            )

                            break_idx = (
                                tensile_points["force_drop_n"].idxmax()
                            )

                            if pd.notna(break_idx) and break_idx > 0:

                                before = tensile_points.loc[
                                    break_idx - 1
                                ]

                                after = tensile_points.loc[
                                    break_idx
                                ]

                                break_start_disp = float(
                                    before["displacement_mm"]
                                )

                                break_end_disp = float(
                                    after["displacement_mm"]
                                )

                                force_drop = float(
                                    after["force_drop_n"]
                                )

                                fc1, fc2 = st.columns(2)

                                with fc1:

                                    st.metric(
                                        "Estimated Failure Region",
                                        (
                                            f"{break_start_disp:.3f}"
                                            f"–"
                                            f"{break_end_disp:.3f} mm"
                                        )
                                    )

                                with fc2:

                                    st.metric(
                                        "Force Drop",
                                        f"{force_drop:.3f} N"
                                    )

                                st.caption(
                                    "Estimated from the largest "
                                    "consecutive force drop in the "
                                    "recovered tensile points. This is "
                                    "not an exact physical break point."
                                )

                            else:

                                st.info(
                                    "A failure region could not be "
                                    "estimated from the available "
                                    "tensile points."
                                )

                        else:

                            st.info(
                                "Not enough recovered tensile points "
                                "are available for failure-region analysis."
                            )

                    except Exception as e:

                        st.warning(
                            "Unable to calculate the estimated "
                            "failure region from the recovered tensile data."
                        )

                        st.exception(e)

                else:

                    st.info(
                        "Recovered tensile-point data is not available."
                    )


                # ------------------------------------------------
                # TENSILE VISUALIZATION
                # ------------------------------------------------

                st.subheader(
                    "Tensile Analysis"
                )


                real_visualization = os.path.join(
                    RESULTS_DIR,
                    (
                        f"EXP_{real_sample['sample_id']}"
                        f"_real_tensile_analysis.png"
                    )
                )


                if os.path.exists(
                    real_visualization
                ):

                    st.image(
                        real_visualization,
                        use_container_width=True,
                    )

                else:

                    st.warning(
                        "Real tensile visualization "
                        "has not been generated yet."
                    )


                # ------------------------------------------------
                # DATA PROVENANCE
                # ------------------------------------------------

                st.subheader(
                    "Data Provenance"
                )


                st.write(
                    f"**Data status:** "
                    f"{real_sample['data_status']}"
                )


                st.write(
                    f"**Source evidence:** "
                    f"{real_sample['source_evidence']}"
                )


                st.write(
                    f"**Prediction source:** "
                    f"{real_sample['prediction_source']}"
                )


                st.caption(
                    str(
                        real_sample[
                            "notes"
                        ]
                    )
                )


        except Exception as e:

            st.error(
                "Unable to load the real experimental dataset."
            )

            st.exception(e)


# ============================================================
# ============================================================
# AI QUALITY PREDICTION
# ============================================================
# ============================================================

with ai_tab:

    st.header(
        "🤖 AI Quality Prediction"
    )

    st.caption(
        "Enter measured PCL properties and allow the "
        "trained model to predict GOOD or POOR quality."
    )


    # ========================================================
    # ADD NEW SAMPLE
    # ========================================================

    with st.expander(
        "➕ Add New PCL Sample",
        expanded=False
    ):

        st.info(
            "Enter the 26 measured properties of a new PCL "
            "sample. The AI model will automatically predict "
            "the quality class and confidence."
        )


        with st.form(
            "add_sample_form"
        ):

            # ------------------------------------------------
            # SAMPLE ID
            # ------------------------------------------------

            st.subheader(
                "Sample Information"
            )


            sample_id_input = st.text_input(
                "Sample ID",
                placeholder="Example: PCL_03"
            )


            st.divider()


            # ------------------------------------------------
            # IMAGE PROPERTIES
            # ------------------------------------------------

            st.subheader(
                "🖼️ Image Properties"
            )


            c1, c2 = st.columns(2)


            with c1:

                new_img_mean_intensity = st.number_input(
                    "Mean Intensity",
                    value=0.0,
                    format="%.6f"
                )

                new_img_intensity_std = st.number_input(
                    "Intensity Std",
                    value=0.0,
                    format="%.6f"
                )

                new_img_texture_contrast = st.number_input(
                    "Texture Contrast",
                    value=0.0,
                    format="%.6f"
                )

                new_img_texture_energy = st.number_input(
                    "Texture Energy",
                    value=0.0,
                    format="%.6f"
                )

                new_img_edge_density = st.number_input(
                    "Edge Density",
                    value=0.0,
                    format="%.6f"
                )


            with c2:

                new_img_fiber_density = st.number_input(
                    "Fiber Density",
                    value=0.0,
                    format="%.6f"
                )

                new_img_orientation_index = st.number_input(
                    "Orientation Index",
                    value=0.0,
                    format="%.6f"
                )

                new_img_porosity_est = st.number_input(
                    "Porosity Estimate",
                    value=0.0,
                    format="%.6f"
                )

                new_img_mean_fiber_width_px = st.number_input(
                    "Mean Fiber Width (px)",
                    value=0.0,
                    format="%.6f"
                )

                new_img_fiber_width_std_px = st.number_input(
                    "Fiber Width Std (px)",
                    value=0.0,
                    format="%.6f"
                )


            st.divider()


            # ------------------------------------------------
            # SPECTROSCOPY
            # ------------------------------------------------

            st.subheader(
                "🔬 Spectroscopy Properties"
            )


            c1, c2 = st.columns(2)


            with c1:

                new_spec_peak1_nm = st.number_input(
                    "Peak 1 Wavelength (nm)",
                    value=0.0,
                    format="%.6f"
                )

                new_spec_peak1_intensity = st.number_input(
                    "Peak 1 Intensity",
                    value=0.0,
                    format="%.6f"
                )

                new_spec_peak2_nm = st.number_input(
                    "Peak 2 Wavelength (nm)",
                    value=0.0,
                    format="%.6f"
                )

                new_spec_peak2_intensity = st.number_input(
                    "Peak 2 Intensity",
                    value=0.0,
                    format="%.6f"
                )

                new_spec_peak3_nm = st.number_input(
                    "Peak 3 Wavelength (nm)",
                    value=0.0,
                    format="%.6f"
                )


            with c2:

                new_spec_peak3_intensity = st.number_input(
                    "Peak 3 Intensity",
                    value=0.0,
                    format="%.6f"
                )

                new_spec_area_norm = st.number_input(
                    "Area Normalized",
                    value=0.0,
                    format="%.6f"
                )

                new_spec_mean = st.number_input(
                    "Spectroscopy Mean",
                    value=0.0,
                    format="%.6f"
                )

                new_spec_std = st.number_input(
                    "Spectroscopy Std",
                    value=0.0,
                    format="%.6f"
                )

                new_spec_baseline_slope = st.number_input(
                    "Baseline Slope",
                    value=0.0,
                    format="%.6f"
                )


            st.divider()


            # ------------------------------------------------
            # MECHANICAL
            # ------------------------------------------------

            st.subheader(
                "⚙️ Mechanical Properties"
            )


            c1, c2 = st.columns(2)


            with c1:

                new_mech_max_force_n = st.number_input(
                    "Maximum Force (N)",
                    value=0.0,
                    format="%.6f"
                )

                new_mech_displacement_at_max_mm = st.number_input(
                    "Displacement at Maximum Force (mm)",
                    value=0.0,
                    format="%.6f"
                )

                new_mech_force_mean_n = st.number_input(
                    "Mean Force (N)",
                    value=0.0,
                    format="%.6f"
                )


            with c2:

                new_mech_force_std_n = st.number_input(
                    "Force Std (N)",
                    value=0.0,
                    format="%.6f"
                )

                new_mech_curve_slope_n_per_mm = st.number_input(
                    "Curve Slope (N/mm)",
                    value=0.0,
                    format="%.6f"
                )

                new_mech_energy_proxy = st.number_input(
                    "Energy Proxy",
                    value=0.0,
                    format="%.6f"
                )


            st.divider()

            # ------------------------------------------------
            # OPTIONAL TENSILE CURVE
            # ------------------------------------------------

            st.subheader(
                "📈 Optional Tensile Failure Analysis"
            )

            st.caption(
                "Upload a tensile CSV to calculate the "
                "sample-specific estimated failure region. "
                "Required columns: displacement_mm and force_n. "
                "time_s is optional."
            )

            tensile_file = st.file_uploader(
                "Upload Tensile CSV",
                type=["csv"],
                key="new_sample_tensile_file"
            )

            submitted = st.form_submit_button(
                "🔬 Add Sample & Predict Quality",
                use_container_width=True
            )


            # =================================================
            # PROCESS NEW SAMPLE
            # =================================================

            if submitted:

                new_id = (
                    sample_id_input.strip()
                )


                if not new_id:

                    st.error(
                        "Please enter a Sample ID."
                    )

                    st.stop()


                existing_ids = (
                    df["sample_id"]
                    .dropna()
                    .astype(str)
                    .tolist()
                )


                if new_id in existing_ids:

                    st.error(
                        f"Sample ID '{new_id}' already exists."
                    )

                    st.stop()


                # ------------------------------------------------
                # CREATE SAMPLE
                # ------------------------------------------------

                new_sample = {

                    "sample_id":
                        new_id,

                    "img_mean_intensity":
                        new_img_mean_intensity,

                    "img_intensity_std":
                        new_img_intensity_std,

                    "img_texture_contrast":
                        new_img_texture_contrast,

                    "img_texture_energy":
                        new_img_texture_energy,

                    "img_edge_density":
                        new_img_edge_density,

                    "img_fiber_density":
                        new_img_fiber_density,

                    "img_orientation_index":
                        new_img_orientation_index,

                    "img_porosity_est":
                        new_img_porosity_est,

                    "img_mean_fiber_width_px":
                        new_img_mean_fiber_width_px,

                    "img_fiber_width_std_px":
                        new_img_fiber_width_std_px,

                    "spec_peak1_nm":
                        new_spec_peak1_nm,

                    "spec_peak1_intensity":
                        new_spec_peak1_intensity,

                    "spec_peak2_nm":
                        new_spec_peak2_nm,

                    "spec_peak2_intensity":
                        new_spec_peak2_intensity,

                    "spec_peak3_nm":
                        new_spec_peak3_nm,

                    "spec_peak3_intensity":
                        new_spec_peak3_intensity,

                    "spec_area_norm":
                        new_spec_area_norm,

                    "spec_mean":
                        new_spec_mean,

                    "spec_std":
                        new_spec_std,

                    "spec_baseline_slope":
                        new_spec_baseline_slope,

                    "mech_max_force_n":
                        new_mech_max_force_n,

                    "mech_displacement_at_max_mm":
                        new_mech_displacement_at_max_mm,

                    "mech_force_mean_n":
                        new_mech_force_mean_n,

                    "mech_force_std_n":
                        new_mech_force_std_n,

                    "mech_curve_slope_n_per_mm":
                        new_mech_curve_slope_n_per_mm,

                    "mech_energy_proxy":
                        new_mech_energy_proxy,
                }


                # ------------------------------------------------
                # MODEL INPUT
                # ------------------------------------------------

                new_X = pd.DataFrame(
                    [[
                        new_sample[
                            column
                        ]
                        for column in feature_columns
                    ]],
                    columns=feature_columns
                )


                # ------------------------------------------------
                # PREDICTION
                # ------------------------------------------------

                new_prediction = (
                    model.predict(
                        new_X
                    )[0]
                )


                new_probabilities = (
                    model.predict_proba(
                        new_X
                    )[0]
                )


                new_confidence = max(
                    new_probabilities
                )


                new_sample[
                    "predicted_quality"
                ] = new_prediction


                new_sample[
                    "prediction_confidence"
                ] = new_confidence

                # ------------------------------------------------
                # OPTIONAL TENSILE ANALYSIS
                # ------------------------------------------------

                tensile_result = None

                if tensile_file is not None:

                    try:

                        tensile_df = pd.read_csv(
                            tensile_file
                        )

                        tensile_result = analyze_tensile_curve(
                            tensile_df
                        )

                    except Exception as e:

                        st.error(
                            "The tensile CSV could not be analyzed. "
                            "Please make sure it contains numeric "
                            "displacement_mm and force_n columns."
                        )

                        st.exception(e)
                        st.stop()

                # ------------------------------------------------
                # SAVE
                # ------------------------------------------------

                df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            [new_sample]
                        ),
                    ],
                    ignore_index=True
                )


                df.to_csv(
                    DATA_PATH,
                    index=False
                )


                st.session_state[
                    "selected_sample"
                ] = new_id

                if tensile_result is not None:
                    st.session_state[
                        "tensile_analysis_results"
                    ][new_id] = tensile_result

                    st.session_state[
                        "tensile_curve_data"
                    ][new_id] = tensile_df.copy()

                    # Persist the uploaded curve so the sample's tensile
                    # analysis remains available after Streamlit restarts.
                    curve_to_save = tensile_df.copy()
                    curve_to_save["sample_id"] = new_id
                    if os.path.exists(TENSILE_CURVES_PATH):
                        existing_curves = pd.read_csv(TENSILE_CURVES_PATH)
                        if "sample_id" in existing_curves.columns:
                            existing_curves = existing_curves[
                                existing_curves["sample_id"].astype(str) != str(new_id)
                            ]
                        combined_curves = pd.concat(
                            [existing_curves, curve_to_save],
                            ignore_index=True
                        )
                    else:
                        combined_curves = curve_to_save
                    combined_curves.to_csv(
                        TENSILE_CURVES_PATH,
                        index=False
                    )

                else:
                    st.session_state[
                        "tensile_analysis_results"
                    ].pop(
                        new_id,
                        None
                    )

                    st.session_state[
                        "tensile_curve_data"
                    ].pop(
                        new_id,
                        None
                    )

                st.success(
                    f"{new_id} added successfully."
                )

                st.rerun()


    # ========================================================
    # EMPTY DATA CHECK
    # ========================================================

    if df.empty:

        st.warning(
            "No PCL samples are available in "
            "actual_pcl_data.csv."
        )

        st.stop()


    # ========================================================
    # SAMPLE SELECTION
    # ========================================================

    st.subheader(
        "Select PCL Sample"
    )


    sample_ids = (
        df["sample_id"]
        .dropna()
        .astype(str)
        .tolist()
    )


    if not sample_ids:

        st.warning(
            "No valid Sample IDs found."
        )

        st.stop()


    if (
        "selected_sample" in st.session_state
        and
        st.session_state[
            "selected_sample"
        ] in sample_ids
    ):

        default_index = (
            sample_ids.index(
                st.session_state[
                    "selected_sample"
                ]
            )
        )

    else:

        default_index = 0


    selected_sample = st.selectbox(
        "PCL Sample ID",
        sample_ids,
        index=default_index,
    )


    st.session_state[
        "selected_sample"
    ] = selected_sample


    # ========================================================
    # SELECTED SAMPLE
    # ========================================================

    sample = df[
        df["sample_id"].astype(str)
        == selected_sample
    ].iloc[0]


    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    X = pd.DataFrame(
        [[
            sample[column]
            for column in feature_columns
        ]],
        columns=feature_columns
    )


    prediction = (
        model.predict(X)[0]
    )


    probabilities = (
        model.predict_proba(X)[0]
    )


    classes = list(
        model.classes_
    )


    probability_dict = dict(
        zip(
            classes,
            probabilities
        )
    )


    confidence = max(
        probabilities
    )


    # ========================================================
    # AI RESULT
    # ========================================================

    st.subheader(
        "AI Quality Assessment"
    )


    m1, m2, m3 = st.columns(3)


    with m1:

        st.metric(
            "Sample ID",
            selected_sample
        )


    with m2:

        st.metric(
            "Predicted Quality",
            prediction
        )


    with m3:

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )


    if prediction == "GOOD":

        st.success(
            f"✓ Quality Result: GOOD — {selected_sample}"
        )

    else:

        st.error(
            f"⚠ Quality Result: POOR — {selected_sample}"
        )


    # ========================================================
    # TENSILE FAILURE ANALYSIS FOR SELECTED SAMPLE
    # ========================================================

    # Read the persisted tensile curve directly on every rerun.
    # This makes the selected-sample analysis independent of Streamlit
    # session state and robust after Cloud restarts/redeployments.
    tensile_result = None
    tensile_curve = None

    if os.path.exists(TENSILE_CURVES_PATH):
        try:
            persisted_tensile = pd.read_csv(
                TENSILE_CURVES_PATH
            )

            if "sample_id" in persisted_tensile.columns:
                selected_curve = persisted_tensile[
                    persisted_tensile["sample_id"].astype(str)
                    == str(selected_sample)
                ].copy()

                if not selected_curve.empty:
                    tensile_curve = selected_curve.drop(
                        columns=["sample_id"]
                    ).copy()

                    tensile_result = analyze_tensile_curve(
                        tensile_curve
                    )

        except Exception:
            tensile_result = None
            tensile_curve = None

    if tensile_result is not None:

        st.subheader(
            "📈 Tensile Failure Analysis"
        )

        tc1, tc2, tc3 = st.columns(3)

        with tc1:

            st.metric(
                "Maximum Force",
                f"{tensile_result['max_force_n']:.3f} N"
            )

        with tc2:

            st.metric(
                "Failure Region",
                (
                    f"{tensile_result['failure_start_mm']:.3f}"
                    f"–"
                    f"{tensile_result['failure_end_mm']:.3f} mm"
                )
            )

        with tc3:

            st.metric(
                "Force Drop",
                f"{tensile_result['force_drop_n']:.3f} N"
            )

        st.caption(
            "Failure region is estimated from the largest "
            "consecutive force drop in the uploaded tensile curve. "
            "It is not an exact physical break point."
        )

        if tensile_curve is not None:

            st.subheader(
                "📊 Tensile Force–Displacement Curve"
            )

            curve_plot = tensile_curve[
                ["displacement_mm", "force_n"]
            ].copy()

            curve_plot["displacement_mm"] = pd.to_numeric(
                curve_plot["displacement_mm"],
                errors="coerce"
            )

            curve_plot["force_n"] = pd.to_numeric(
                curve_plot["force_n"],
                errors="coerce"
            )

            curve_plot = curve_plot.dropna().sort_values(
                "displacement_mm"
            )

            fig, ax = plt.subplots(
                figsize=(10, 5)
            )

            ax.plot(
                curve_plot["displacement_mm"],
                curve_plot["force_n"],
                marker="o",
                markersize=3,
                linewidth=2,
                label="Tensile curve"
            )

            max_idx = curve_plot["force_n"].idxmax()
            max_point = curve_plot.loc[max_idx]

            ax.scatter(
                max_point["displacement_mm"],
                max_point["force_n"],
                s=70,
                zorder=5,
                label="Maximum force"
            )

            ax.axvspan(
                tensile_result["failure_start_mm"],
                tensile_result["failure_end_mm"],
                alpha=0.20,
                label="Estimated failure region"
            )

            ax.set_title(
                f"{selected_sample} — Tensile Analysis"
            )
            ax.set_xlabel(
                "Displacement (mm)"
            )
            ax.set_ylabel(
                "Force (N)"
            )
            ax.grid(
                True,
                alpha=0.25
            )
            ax.legend()
            fig.tight_layout()

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)

    # ========================================================
    # PROPERTY TABS
    # ========================================================

    st.subheader(
        "Measured PCL Properties"
    )


    tab_image, tab_spec, tab_mech = st.tabs(
        [
            "🖼️ Image Properties",
            "🔬 Spectroscopy",
            "⚙️ Mechanical Properties",
        ]
    )


    # ========================================================
    # PROPERTY DISPLAY FUNCTION
    # ========================================================

    def show_properties(
        properties
    ):

        for i in range(
            0,
            len(properties),
            2
        ):

            cols = st.columns(2)


            for j, feature in enumerate(
                properties[
                    i:i + 2
                ]
            ):

                value = sample[
                    feature
                ]


                if pd.isna(value):

                    value_text = (
                        "Not available"
                    )

                else:

                    value_text = (
                        f"{float(value):.6f}"
                    )


                with cols[j]:

                    st.metric(
                        display_names.get(
                            feature,
                            feature
                        ),
                        value_text
                    )


    # ========================================================
    # IMAGE TAB
    # ========================================================

    with tab_image:

        show_properties(
            image_features
        )


    # ========================================================
    # SPECTROSCOPY TAB
    # ========================================================

    with tab_spec:

        show_properties(
            spectroscopy_features
        )


    # ========================================================
    # MECHANICAL TAB
    # ========================================================

    with tab_mech:

        show_properties(
            mechanical_features
        )


    # ========================================================
    # AI VISUAL ANALYSIS
    # ========================================================

    st.subheader(
        "AI Visual Analysis"
    )


    st.caption(
        f"Sample-specific analysis for {selected_sample}"
    )


    # ========================================================
    # LOAD TRAINING DATA
    # ========================================================

    try:

        train_df = pd.read_csv(
            TRAIN_PATH
        )


        train_numeric = (
            train_df[
                feature_columns
            ]
            .apply(
                pd.to_numeric,
                errors="coerce"
            )
        )


        sample_numeric = (
            sample[
                feature_columns
            ]
            .apply(
                pd.to_numeric,
                errors="coerce"
            )
        )


        train_min = (
            train_numeric.min()
        )


        train_max = (
            train_numeric.max()
        )


        train_range = (
            train_max
            - train_min
        )


        train_range = (
            train_range.replace(
                0,
                1
            )
        )


        normalized_values = (
            (
                sample_numeric
                - train_min
            )
            / train_range
        )


        normalized_values = (
            normalized_values
            .fillna(0)
            .clip(0, 1)
        )


    except Exception as e:

        st.error(
            "Unable to prepare the AI visualization."
        )

        st.exception(e)

        st.stop()


    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    importance_df = pd.DataFrame(
        {
            "feature":
                feature_columns,

            "importance":
                model.feature_importances_,
        }
    )


    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=True
        )
    )


    top_features = (
        importance_df
        .tail(10)
    )


    # ========================================================
    # CLASS PROBABILITIES
    # ========================================================

    good_probability = (
        probability_dict.get(
            "GOOD",
            0
        )
    )


    poor_probability = (
        probability_dict.get(
            "POOR",
            0
        )
    )


    # ========================================================
    # CREATE FIGURE
    # ========================================================

    fig = plt.figure(
        figsize=(16, 9)
    )


    fig.suptitle(
        "PCL Non-Woven Fabric Quality Detection",
        fontsize=20,
        fontweight="bold"
    )


    # ========================================================
    # HEATMAP
    # ========================================================

    ax1 = plt.subplot(
        2,
        2,
        1
    )


    heatmap_data = (
        normalized_values
        .values
        .reshape(-1, 1)
    )


    heatmap = ax1.imshow(
        heatmap_data,
        aspect="auto",
        interpolation="nearest"
    )


    ax1.set_title(
        "PCL Sample Property Heatmap",
        fontweight="bold"
    )


    ax1.set_xlabel(
        "Selected Sample"
    )


    ax1.set_ylabel(
        "Measured Property"
    )


    ax1.set_xticks(
        [0]
    )


    ax1.set_xticklabels(
        [selected_sample]
    )


    ax1.set_yticks(
        range(
            len(feature_columns)
        )
    )


    ax1.set_yticklabels(
        feature_columns,
        fontsize=7
    )


    fig.colorbar(
        heatmap,
        ax=ax1,
        fraction=0.046,
        pad=0.04,
        label="Normalized Property Value"
    )


    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    ax2 = plt.subplot(
        2,
        2,
        2
    )


    ax2.barh(
        top_features["feature"],
        top_features["importance"]
    )


    ax2.set_title(
        "Important PCL Features",
        fontweight="bold"
    )


    ax2.set_xlabel(
        "Random Forest Feature Importance"
    )


    # ========================================================
    # AI PREDICTION
    # ========================================================

    ax3 = plt.subplot(
        2,
        2,
        3
    )


    probability_values = [
        good_probability,
        poor_probability
    ]


    bars = ax3.bar(
        [
            "GOOD",
            "POOR"
        ],
        probability_values
    )


    ax3.set_ylim(
        0,
        1
    )


    ax3.set_ylabel(
        "Prediction Probability"
    )


    ax3.set_title(
        "AI Quality Prediction",
        fontweight="bold"
    )


    for bar, value in zip(
        bars,
        probability_values
    ):

        ax3.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 0.025,
            f"{value * 100:.1f}%",
            ha="center",
            fontweight="bold"
        )


    # ========================================================
    # PROPERTY PROFILE
    # ========================================================

    ax4 = plt.subplot(
        2,
        2,
        4
    )


    ax4.plot(
        range(
            1,
            len(feature_columns) + 1
        ),
        normalized_values.values,
        marker="o"
    )


    ax4.set_title(
        "PCL Sample Property Profile",
        fontweight="bold"
    )


    ax4.set_xlabel(
        "Measured Property Number"
    )


    ax4.set_ylabel(
        "Normalized Value"
    )


    ax4.set_ylim(
        0,
        1
    )


    ax4.set_xticks(
        range(
            1,
            len(feature_columns) + 1
        )
    )


    ax4.tick_params(
        axis="x",
        labelsize=7
    )


    ax4.grid(
        True,
        alpha=0.3
    )


    # ========================================================
    # FOOTER
    # ========================================================

    fig.text(
        0.5,
        0.015,
        (
            f"Sample ID: {selected_sample}"
            f"   |   "
            f"Predicted Quality: {prediction}"
            f"   |   "
            f"Confidence: {confidence * 100:.2f}%"
        ),
        ha="center",
        fontsize=13,
        fontweight="bold"
    )


    plt.tight_layout(
        rect=[
            0,
            0.045,
            1,
            0.95
        ]
    )


    # ========================================================
    # SAVE VISUALIZATION
    # ========================================================

    visualization_path = os.path.join(
        RESULTS_DIR,
        f"{selected_sample}_analysis.png"
    )


    fig.savefig(
        visualization_path,
        dpi=150,
        bbox_inches="tight"
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    st.image(
        visualization_path,
        use_container_width=True
    )


    plt.close(
        fig
    )


# ============================================================
# APPLICATION FOOTER
# ============================================================

st.divider()

st.caption(
    "PCL Non-Woven Fabric Quality Detection • "
    "AI-assisted material quality analysis"
)
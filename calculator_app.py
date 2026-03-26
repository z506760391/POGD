import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ===========================================================
# Page Configuration
# ===========================================================
st.set_page_config(
    page_title="POGD Risk Prediction Calculator",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ===========================================================
# Helper Functions (copied and simplified from the main script)
# ===========================================================
def agg_pred(bag_mods, X, nm):
    pl = []
    for bd in bag_mods:
        m = bd.get(nm)
        if m:
            try:
                pl.append(m.predict_proba(X)[:, 1])
            except Exception:
                pass
    return np.mean(pl, axis=0) if pl else np.zeros(len(X))


def wsv(bag_mods, X, tops, weights_dict):
    pl, ws = [], []
    for n in tops:
        w = weights_dict.get(n, 0.)
        ws.append(0. if (isinstance(w, float) and np.isnan(w)) else float(w))
        pl.append(agg_pred(bag_mods, X, n))
    ws = np.array(ws, dtype=float)
    ws = ws / ws.sum() if ws.sum() > 0 else np.ones(len(ws)) / len(ws)
    return np.dot(ws, np.array(pl))


# ===========================================================
# Load Model Components
# ===========================================================
@st.cache_resource
def load_model_components():
    """
    Load all necessary model components.
    Uses Streamlit caching to avoid reloading on every interaction.
    """
    path = 'output/ensemble_model_components.pkl'
    if not os.path.exists(path):
        st.error(f"Error: Model file '{path}' not found. Please run the main analysis script `ml_pogd_final_v5_1 - 副本.py` first to generate the model file.")
        return None
    try:
        components = joblib.load(path)
        return components
    except Exception as e:
        st.error(f"Error loading model file: {e}")
        return None


components = load_model_components()

# ===========================================================
# Main Application Interface
# ===========================================================
st.title("⚕️ POGD Risk Prediction Calculator for Postoperative Gastric Dysfunction")
st.markdown("---")
st.markdown("""
This tool calculates a patient's risk of developing postoperative gastric dysfunction (POGD)
based on a published prediction model. Please enter the patient's clinical indicators below.
"""
)

if components:
    # Extract required objects from loaded components
    selected_feature_list = components.get('selected_feature_list', [])
    full_feature_list = components.get('full_feature_list', [])
    scaler = components.get('scaler')
    label_encoders = components.get('label_encoders', {})
    imputation_values = components.get('imputation_values', {})
    bagged_models = components.get('bagged_models')
    top_models = components.get('top_models_for_ensemble')
    ensemble_weights = components.get('ensemble_weights')

    if not all([selected_feature_list, full_feature_list, scaler, bagged_models, top_models, ensemble_weights]):
        st.error("The model file `ensemble_model_components.pkl` is corrupted or missing key components. Please re-run the main analysis script.")
        st.stop()

    st.sidebar.header("Patient Clinical Information Input")

    input_data = {}

    # --- Continuous Variable Inputs ---
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=60, help="Patient must be an adult (≥18 years old)")
bmi = st.sidebar.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=50.0, value=22.0, format="%.2f", help="Range: 10-50")
blood = st.sidebar.number_input("Intraoperative Blood Loss (ml)", min_value=0, max_value=5000, value=100, help="Range: 0-5000 ml")

    # --- Categorical Variable Inputs ---
    node12_options = {'Yes': 1, 'No': 0}
    node12_selection = st.sidebar.selectbox("Harvested Pathological Lymph Nodes ≥12 (node12)", options=list(node12_options.keys()), help="Harvested pathological lymph nodes ≥12")

    tmn_options = {
        "Stage 0 (Value 0)": 0,
        "Stage 0 (Value 1)": 1,
        "Stage Ⅰ (T1-2N0M0)": 2,
        "Stage Ⅱ (T3-4N0M0)": 3,
        "Stage Ⅲ (T1-4N1-2M0)": 4,
        "Stage Ⅳ (T1-4N0-2M1)": 5
    }
    tmn_selection = st.sidebar.selectbox("TMN Staging (TMN)", options=list(tmn_options.keys()), help="Tumour characteristics")

    add_options = {'Yes': 1, 'No': 0}
    add_selection = st.sidebar.selectbox("Additional Organ Resection (Add)", options=list(add_options.keys()), help="Additional organ resection")

    dixon_miles_options = {'Abdominoperineal Resection (APR)': 1, 'Low Anterior Resection (LAR)': 0}
    dixon_miles_selection = st.sidebar.selectbox("Surgical Procedure (DixonorRMiles)", options=list(dixon_miles_options.keys()))

    distant_group_options = {'≥7 cm': 1, '<7 cm': 0}
    distant_group_selection = st.sidebar.selectbox("Tumour Distance from Anal Verge (distant_group)", options=list(distant_group_options.keys()))

    nat_f_options = {'Yes': 1, 'No': 0}
    nat_f_selection = st.sidebar.selectbox("Neoadjuvant Therapy (NAT_f)", options=list(nat_f_options.keys()), help="Neoadjuvant therapy")

    # Map user-selected text options to numeric values required by the model
    input_data = {
        'Age': age,
        'BMI': bmi,
        'blood': blood,
        'node12': node12_options[node12_selection],
        'TMN': tmn_options[tmn_selection],
        'Add': add_options[add_selection],
        'DixonorRMiles': dixon_miles_options[dixon_miles_selection],
        'distant_group': distant_group_options[distant_group_selection],
        'NAT_f': nat_f_options[nat_f_selection],
    }

    # "Calculate Risk" button
    if st.sidebar.button("Calculate Risk", use_container_width=True, type="primary"):

        # --- Data Preprocessing ---
        # 1. Place user-input selected features into a DataFrame
        input_df = pd.DataFrame([input_data])

        # 2. Build a complete DataFrame containing all original features, filling missing ones with imputation values
        full_df_data = {}
        for col in full_feature_list:
            if col in input_df.columns:
                full_df_data[col] = input_df[col].iloc[0]
            else:
                full_df_data[col] = imputation_values.get(col)
        full_df = pd.DataFrame([full_df_data])

        # 3. Preprocess the complete DataFrame
        # 3.1 Encode categorical variables
        for col, le in label_encoders.items():
            if col in full_df.columns:
                # Use transform with fallback to handle unseen categories
                full_df[col] = full_df[col].apply(lambda x: le.transform([x])[0] if str(x) in le.classes_ else le.transform(['__unk__'])[0])

        # 3.2 Ensure column order matches training order exactly (safety measure)
        full_df = full_df[full_feature_list]

        # 4. Scale the complete DataFrame
        try:
            scaled_full_df = pd.DataFrame(scaler.transform(full_df), columns=full_feature_list)

            # 5. Select only the final predictors required by the model from the scaled full data
            model_input_df = scaled_full_df[selected_feature_list]

            # --- Run Prediction ---
            prediction_proba = wsv(bagged_models, model_input_df, top_models, ensemble_weights)
            risk_percentage = prediction_proba[0] * 100

            # --- Display Results ---
            st.header("Prediction Results")
            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    label="POGD Risk Probability",
                    value=f"{risk_percentage:.2f} %"
                )

            with col2:
                if risk_percentage >= 50:
                    st.error("Conclusion: High Risk")
                elif risk_percentage >= 30:
                    st.warning("Conclusion: Moderate Risk")
                else:
                    st.success("Conclusion: Low Risk")

            st.progress(risk_percentage / 100)

            with st.expander("View Detailed Input Data (Final Predictors)"):
                st.dataframe(input_df)
            with st.expander("View Complete Data Built for Preprocessing"):
                st.dataframe(full_df)
            with st.expander("View Final Model Input Data (After Scaling)"):
                st.dataframe(model_input_df)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.info("Please ensure all input values are valid.")

else:
    st.warning("Model components could not be loaded. The calculator is unavailable.")

st.markdown("---")
st.info("""
**Disclaimer:** This calculator is intended for academic research and demonstration purposes only.
Its results should not be used as the sole basis for clinical decision-making.
All clinical decisions should be made by qualified medical professionals.
""")
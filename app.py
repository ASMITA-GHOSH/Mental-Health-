import streamlit as st
import joblib
import pandas as pd
import numpy as np

# -------------------------------------------------------------
# 1. PAGE CONFIG (MUST BE THE FIRST STREAMLIT COMMAND)
# -------------------------------------------------------------
st.set_page_config(
    page_title="Tech Industry Mental Health Risk Predictor",
    layout="wide"  # Use "wide" layout for better chart visibility
)

# -------------------------------------------------------------
# 2. MODEL LOADING
# -------------------------------------------------------------
MODEL_FILE = 'mock_mental_health_model.joblib'
try:
    # Load the mock model (trained using Logistic Regression)
    model = joblib.load(MODEL_FILE)
except FileNotFoundError:
    st.error(f"Error: Model file '{MODEL_FILE}' not found. Please ensure it is in the same directory.")
    st.stop()

# -------------------------------------------------------------
# 3. SIDEBAR FOR INFO (Phase 4)
# -------------------------------------------------------------
with st.sidebar:
    st.title("About This Tool")
    st.markdown("""
    This application is a *proof-of-concept* for predicting mental health risk in the tech industry.

    *⚠ Disclaimer:* This model was trained on an *extremely small dataset (6 rows)* and is *not suitable for clinical use or real HR decision-making.* The results are purely illustrative of the process.
    """)
    st.header("Model Features Used")
    st.markdown("""
    The prediction is based on three features:
    - *Employer's Mental Health Importance* (1-5 scale)
    - *Family History* of Mental Illness (Yes/No/I Don't Know)
    - *Primary Role in Tech/IT* (Yes/No)
    """)

# -------------------------------------------------------------
# 4. MAIN APPLICATION INTERFACE
# -------------------------------------------------------------
st.title("HR Well-being Tool: Mental Health Risk Assessment 🧠💻")
st.markdown("Please answer the following questions to receive a *mock* risk assessment.")

# --- Questionnaire Input Form (Phase 4) ---
with st.form("risk_assessment_form"):
    st.header("Input Questionnaire")

    # 1. Employer Importance (Ordinal)
    employer_importance = st.slider(
        "1. How much importance does your employer place on mental health (1 = Very Low, 5 = Very High)?",
        min_value=1, max_value=5, value=3, key="s1"
    )

    # 2. Family History (Categorical)
    family_history = st.radio(
        "2. Do you have a family history of mental illness?",
        options=["Yes", "No", "I don't know"],
        index=2, key="r2"
    )

    # 3. Role in Tech (Binary)
    is_tech_role = st.selectbox(
        "3. Is your primary role within your company related to tech/IT?",
        options=["Yes", "No"], key="s3"
    )

    submitted = st.form_submit_button("Predict Risk")

# -------------------------------------------------------------
# 5. PREDICTION AND CHART LOGIC
# -------------------------------------------------------------
if submitted:
    # 5.1. Data Preparation (Matching Training Preprocessing)
    
    # Feature 1 (Employer Importance): Already numerical (1-5)

    # Feature 2 (Family History): Encode to 0, 1, or 2
    # Mapping used in training: {'Yes': 2, 'No': 0, "I don't know": 1}
    family_history_map = {'Yes': 2, 'No': 0, "I don't know": 1}
    family_history_encoded = family_history_map[family_history]

    # Feature 3 (Role in Tech): Encode to 0.0 or 1.0
    is_tech_role_encoded = 1.0 if is_tech_role == "Yes" else 0.0

    # Create the input DataFrame, ensuring column order matches the training data
    input_data = pd.DataFrame([[
        employer_importance,
        family_history_encoded,
        is_tech_role_encoded
    ]], columns=model.feature_names_in_)

    # 5.2. Prediction and Probability
    risk_prediction = model.predict(input_data)[0]
    # Predict_proba returns [P(class 0), P(class 1)], we want P(class 1)
    risk_probability = model.predict_proba(input_data)[0][1]

    # 5.3. Display Prediction
    st.header("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        # Risk Status
        if risk_prediction == 1:
            st.error(f"## AT RISK")
            st.markdown("⚠ *High Risk* based on the model's current factors.")
        else:
            st.success(f"## NOT AT RISK")
            st.markdown("✅ *Lower Risk* based on the model's current factors.")

    with col2:
        # Probability Score
        st.subheader("Confidence Score (P = At Risk)")
        st.metric(label="Risk Probability", value=f"{risk_probability*100:.2f}%")
        st.progress(risk_probability, text="Confidence Level")
        
    st.markdown("---")

    # 5.4. Feature Importance Chart (Graph)
    st.subheader("Model Explanation: Feature Influence")
    st.markdown("This chart shows the weight (coefficient) each factor had on the prediction. Features with a *positive score* increase the likelihood of being 'At Risk'.")

    # Extract coefficients and feature names from the loaded model
    coefficients = model.coef_[0]
    feature_names = model.feature_names_in_

    # Create a DataFrame for plotting and rename features for readability
    feature_map = {
        'overall_how_much_importance_does_your_employer_place_on_mental_health': 'Employer MH Importance (Higher = Less Risk)',
        'do_you_have_a_family_history_of_mental_illness': 'Family History (Higher = More Risk)',
        'is_your_primary_role_within_your_company_related_to_tech_it': 'Primary Role in Tech/IT'
    }

    coef_df = pd.DataFrame({
        'Feature': [feature_map.get(name, name) for name in feature_names],
        'Coefficient': coefficients
    }).sort_values(by='Coefficient', ascending=True) # Sort ascending for a logical horizontal bar chart

    # Display chart using Altair (which st.bar_chart uses behind the scenes) or st.bar_chart
    st.bar_chart(
        coef_df.set_index('Feature'),
        use_container_width=True,
        height=350
    )

    st.markdown("""
    ---
    Always consult a medical or mental health professional for a true diagnosis or assessment.
    """)
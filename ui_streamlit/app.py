import streamlit as st
from utils import predict_combined



st.set_page_config(page_title="Readmission Risk AI", layout="centered")

st.title("Diabetes Readmission Predictor")

with st.form("patient_form"):
    age = st.slider("Age", 18, 100, 60)
    gender = st.selectbox("Gender", ["male", "female"])
    race = st.selectbox("Race", ["white", "black", "asian", "hispanic", "other"])
    complaint = st.text_area("Chief Complaint (optional)")

    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = {
        "age": age,
        "gender": gender.lower(),
        "race": race.lower(),
        "chief_complaint": complaint if complaint.strip() else None
    }

    # Call combined endpoint
    with st.spinner("Analyzing patient data..."):
        result = predict_combined(input_data)

    # Display readmission risk
    st.metric("📊 Readmission Risk", f"{result['readmission_risk']*100:.1f}%")
    
    # Display NLP-predicted diagnoses (if complaint was provided)
    if result.get('predicted_diagnoses') and len(result['predicted_diagnoses']) > 0:
        st.subheader("🩺 Predicted Diagnoses (from Chief Complaint):")
        for diag in result['predicted_diagnoses']:
            st.write(f"• **{diag['diagnosis']}** ({diag['probability']*100:.1f}% confidence)")

    # Display SHAP risk factors
    st.subheader("📈 Top Risk Factors (SHAP):")
    for k, v in result['risk_factors'].items():
        # Format with color based on positive/negative contribution
        color = "🔴" if v > 0 else "🟢"
        st.write(f"{color} **{k}**: {v:.3f}")

import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("breast_cancer_model.pkl")

# Page config
st.set_page_config(page_title="Breast Cancer Prediction", page_icon="🩺", layout="wide")

# Title
st.title("🩺 Breast Cancer Prediction")
st.markdown("Enter tumor features below and predict whether it is **Malignant (M)** or **Benign (B)**.")

# Feature groups
mean_features = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave points_mean", "symmetry_mean", "fractal_dimension_mean"
]

se_features = [
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave points_se", "symmetry_se",
    "fractal_dimension_se"
]

worst_features = [
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
]

# Layout: 3 columns
col1, col2, col3 = st.columns(3)
user_input = {}

# Column 1: Mean values
with col1:
    st.subheader("📊 Mean Values")
    for feature in mean_features:
        user_input[feature] = st.number_input(feature, format="%.5f")

# Column 2: SE values
with col2:
    st.subheader("📊 SE Values")
    for feature in se_features:
        user_input[feature] = st.number_input(feature, format="%.5f")

# Column 3: Worst values
with col3:
    st.subheader("📊 Worst Values")
    for feature in worst_features:
        user_input[feature] = st.number_input(feature,  format="%.5f")

# Prediction button
if st.button("🔍 Predict", use_container_width=True):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    result = "🛑 Malignant (Cancer Detected)" if prediction == 1 else "✅ Benign (No Cancer)"

    # Show result
    st.markdown("---")
    st.subheader("📌 Prediction Result")
    st.success(result)

    # Show probability
    proba = model.predict_proba(input_df)[0]
    st.write(f"Confidence → Benign: {proba[0]:.2f}, Malignant: {proba[1]:.2f}")

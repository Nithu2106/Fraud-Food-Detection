import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Food Fraud Detection", layout="centered")

# ---------------- TITLE ----------------
st.markdown("<h1 style='color:#2E86C1;'>🍎 Smart Food Adulteration Detection</h1>", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("perfect_food_adulteration_dataset_500_unique.csv")
    df.columns = df.columns.str.strip().str.lower()
    return df

df = load_data()

# ---------------- RENAME ----------------
rename_map = {}
for col in df.columns:
    if "product" in col:
        rename_map[col] = "product_name"
    elif "brand" in col:
        rename_map[col] = "brand"
    elif "detect" in col:
        rename_map[col] = "detection_method"
    elif "severity" in col:
        rename_map[col] = "severity"
    elif "adulterant" in col:
        rename_map[col] = "adulterant"
    elif "risk" in col:
        rename_map[col] = "health_risk"
    elif "fraud" in col:
        rename_map[col] = "fraud_food_type"

df.rename(columns=rename_map, inplace=True)

# ---------------- CLEAN ----------------
df = df.dropna()

required_cols = [
    'product_name', 'brand', 'detection_method',
    'severity', 'adulterant', 'fraud_food_type', 'health_risk'
]

df = df[required_cols]
original_df = df.copy()

# ---------------- ENCODE ----------------
label_encoders = {}
encoded_df = df.copy()

for col in encoded_df.columns:
    le = LabelEncoder()
    encoded_df[col] = le.fit_transform(encoded_df[col])
    label_encoders[col] = le

# ---------------- MODEL ----------------
model = DiscreteBayesianNetwork([
    ('product_name', 'adulterant'),
    ('brand', 'adulterant'),
    ('detection_method', 'adulterant'),

    ('adulterant', 'fraud_food_type'),
    ('product_name', 'fraud_food_type'),

    ('fraud_food_type', 'health_risk'),
    ('severity', 'health_risk')
])

model.fit(encoded_df, estimator=MaximumLikelihoodEstimator)
inference = VariableElimination(model)

# ---------------- UI ----------------
st.sidebar.header("Enter Food Details")

def get_input(col):
    return st.sidebar.selectbox(col, label_encoders[col].classes_)

user_input_raw = {
    'product_name': get_input('product_name'),
    'brand': get_input('brand'),
    'detection_method': get_input('detection_method'),
    'severity': get_input('severity')
}

# Encode
user_input = {
    col: label_encoders[col].transform([val])[0]
    for col, val in user_input_raw.items()
}

# Safe evidence
valid_nodes = set(model.nodes())
safe_evidence = {k: v for k, v in user_input.items() if k in valid_nodes}

# ---------------- PREDICT ----------------
if st.sidebar.button("Predict"):

    st.subheader("🤖 vs 📊 Comparison")

    # STEP 1
    adulterant_result = inference.query(
        variables=['adulterant'],
        evidence=safe_evidence
    )
    adulterant_idx = adulterant_result.values.argmax()

    # STEP 2
    evidence_2 = safe_evidence.copy()
    evidence_2['adulterant'] = adulterant_idx

    # STEP 3
    fraud_result = inference.query(
        variables=['fraud_food_type'],
        evidence=evidence_2
    )
    fraud_idx = fraud_result.values.argmax()

    # STEP 4
    evidence_3 = evidence_2.copy()
    evidence_3['fraud_food_type'] = fraud_idx

    # STEP 5
    risk_result = inference.query(
        variables=['health_risk'],
        evidence=evidence_3
    )

    # DECODE
    adulterant_label = label_encoders['adulterant'].inverse_transform([adulterant_idx])[0]
    fraud_label = label_encoders['fraud_food_type'].inverse_transform([fraud_idx])[0]
    risk_labels = label_encoders['health_risk'].inverse_transform(range(len(risk_result.values)))
    predicted_risk = risk_labels[risk_result.values.argmax()]

    # ---------------- DATASET MATCH ----------------
    filtered = original_df.copy()

    for col, val in user_input_raw.items():
        if col in filtered.columns:
            filtered = filtered[filtered[col] == val]

    # ---------------- DISPLAY ----------------
    col1, col2 = st.columns(2)

    with col1:
        st.info("🤖 Predicted Result")
        st.write(f"Adulterant: {adulterant_label}")
        st.write(f"Fraud Type: {fraud_label}")
        st.write(f"Health Risk: {predicted_risk}")

    with col2:
        st.info("📊 Dataset Result")

        if not filtered.empty:
            st.success("Match found")
            st.write(f"Adulterant: {filtered['adulterant'].mode()[0]}")
            st.write(f"Fraud Type: {filtered['fraud_food_type'].mode()[0]}")
            st.write(f"Health Risk: {filtered['health_risk'].mode()[0]}")
        else:
            st.warning("No exact match in dataset")

    # ---------------- GRAPH ----------------
    st.subheader("📊 Health Risk Distribution")

    fig, ax = plt.subplots()
    ax.bar(risk_labels, risk_result.values)
    ax.set_title("Predicted Risk Probability")
    ax.set_xlabel("Health Risk")
    ax.set_ylabel("Probability")
    st.pyplot(fig)

    # ---------------- WARNING ----------------
    if filtered.empty:
        st.warning("⚠️ Input not found in dataset. Prediction is probability-based.")

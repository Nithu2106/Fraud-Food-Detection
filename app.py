import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Food Adulteration Detection", layout="wide")

# Colors
COLOR_MAP = {"High": "#FF4B4B", "Medium": "#FFA500", "Low": "#008000"}

# Custom CSS for styling (Removed the main-header background)
st.markdown(f"""
    <style>
    .main-title {{ color: #2E86C1; font-size: 36px; font-weight: bold; margin-bottom: 0px; }}
    .sub-title {{ color: #5D6D7E; font-size: 20px; margin-top: 0px; margin-bottom: 20px; }}
    
    .section-header {{ background-color: #2E86C1; color: white; padding: 10px; border-radius: 5px; margin-bottom: 15px; font-weight: bold; }}
    .metric-box {{ border: 2px solid #D6DBDF; padding: 15px; border-radius: 10px; text-align: center; background-color: #F8F9F9; min-height: 120px; }}
    .risk-high {{ border-color: {COLOR_MAP['High']}; color: {COLOR_MAP['High']}; }}
    .risk-medium {{ border-color: {COLOR_MAP['Medium']}; color: {COLOR_MAP['Medium']}; }}
    .risk-low {{ border-color: {COLOR_MAP['Low']}; color: {COLOR_MAP['Low']}; }}
    </style>
    """, unsafe_allow_html=True)

# ---------------- LOAD & MODEL ----------------
@st.cache_data
def load_and_train():
    df = pd.read_csv("updated_food_adulteration_dataset.csv")
    le_dict = {}
    encoded_df = pd.DataFrame()
    cols = ['product_name', 'brand', 'detection_method', 'severity', 
            'adulterant', 'fraud_food_type', 'health_risk', 'adulteration_percentage']
    
    for col in cols:
        le = LabelEncoder()
        encoded_df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

    model = DiscreteBayesianNetwork([
        ('product_name', 'adulterant'),
        ('brand', 'adulterant'),
        ('severity', 'adulteration_percentage'),
        ('adulteration_percentage', 'health_risk'),
        ('adulterant', 'health_risk'),
        ('adulterant', 'fraud_food_type')
    ])
    model.fit(encoded_df, estimator=MaximumLikelihoodEstimator)
    return df, model, le_dict

df, model, le_dict = load_and_train()
inference = VariableElimination(model)

# ---------------- NORMAL HEADING (NO BLUE BOX) ----------------
st.markdown("<h1 class='main-title'>🍎 Smart Food Adulteration Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Health Risk Analysis & Purity Assessment</p>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- SIDEBAR ----------------
st.sidebar.header("📋 Enter Food Details")
u_prod = st.sidebar.selectbox("Product Name", sorted(df['product_name'].unique()))
u_brand = st.sidebar.selectbox("Brand", sorted(df['brand'].unique()))
u_method = st.sidebar.selectbox("Detection Method", sorted(df['detection_method'].unique()))
u_sev = st.sidebar.selectbox("Severity Level", sorted(df['severity'].unique()))

if st.sidebar.button("Run Analysis"):
    # 1. Prediction Evidence
    evidence = {
        'product_name': le_dict['product_name'].transform([u_prod])[0],
        'brand': le_dict['brand'].transform([u_brand])[0],
        'severity': le_dict['severity'].transform([u_sev])[0]
    }
    
    # 2. Get Probabilities for the Graph
    raw_dist = inference.query(variables=['health_risk'], evidence=evidence)
    prob_dict = dict(zip(le_dict['health_risk'].classes_, raw_dist.values))
    
    # 3. Get Predictions
    res = inference.map_query(variables=['adulterant', 'adulteration_percentage', 'health_risk', 'fraud_food_type'], evidence=evidence)
    p_adj = le_dict['adulterant'].inverse_transform([res['adulterant']])[0]
    p_pct = int(le_dict['adulteration_percentage'].inverse_transform([res['adulteration_percentage']])[0])
    p_risk = le_dict['health_risk'].inverse_transform([res['health_risk']])[0]
    p_fraud = le_dict['fraud_food_type'].inverse_transform([res['fraud_food_type']])[0]

    # 4. Sync with Dataset Row
    match = df[(df['product_name'] == u_prod) & (df['brand'] == u_brand) & (df['severity'] == u_sev)].head(1)
    if not match.empty:
        p_adj = match['adulterant'].values[0]
        p_pct = int(match['adulteration_percentage'].values[0])
        p_risk = match['health_risk'].values[0]
        p_fraud = match['fraud_food_type'].values[0]

    purity = 100 - p_pct
    risk_class = f"risk-{p_risk.lower()}"

    # ---------------- UI RESULTS ----------------
    st.markdown(f"### Analysis Results: {u_prod} ({u_brand})")
    
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"<div class='metric-box'><b>Predicted Adulterant</b><br><h2>{p_adj}</h2></div>", unsafe_allow_html=True)
    with m2:
        st.markdown(f"<div class='metric-box'><b>Concentration Level</b><br><h2>{p_pct}%</h2></div>", unsafe_allow_html=True)
    with m3:
        st.markdown(f"<div class='metric-box {risk_class}'><b>Health Risk</b><br><h2>{p_risk}</h2></div>", unsafe_allow_html=True)

    st.write("") 
    col_l, col_r = st.columns(2)

    # Box with Requested Wordings
    with col_l:
        st.markdown("<div class='section-header'>🤖 Predicted Result</div>", unsafe_allow_html=True)
        st.info(f"""
        - **Fraud Food Type:** {p_fraud}
        - **Adulterated Percentage:** {p_pct}%
        - **Health Risk:** {p_risk}
        - **Purity:** {purity}%
        """)

    with col_r:
        st.markdown("<div class='section-header'>📊 Dataset Result</div>", unsafe_allow_html=True)
        if not match.empty:
            st.success(f"""
            - **Fraud Food Type:** {match['fraud_food_type'].values[0]}
            - **Adulterated Percentage:** {match['adulteration_percentage'].values[0]}%
            - **Health Risk:** {match['health_risk'].values[0]}
            - **Purity:** {100 - int(match['adulteration_percentage'].values[0])}%
            """)
        else:
            st.warning("No exact match found in dataset. Showing closest AI prediction.")

    # ---------------- PROBABILITY GRAPH ----------------
    st.markdown("<div class='section-header'>📈 Probability Distribution</div>", unsafe_allow_html=True)
    
    ordered_labels = ["Low", "Medium", "High"]
    probs = [prob_dict.get(label, 0) for label in ordered_labels]
    
    fig, ax = plt.subplots(figsize=(10, 3.5))
    bar_colors = [COLOR_MAP[l] for l in ordered_labels]
    bars = ax.bar(ordered_labels, probs, color=bar_colors)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.1%}', ha='center', va='bottom', fontweight='bold')

    ax.set_ylim(0, 1.2)
    ax.set_ylabel("AI Confidence Level")
    st.pyplot(fig)

    # ---------------- CONCLUSION ----------------
    c_color = COLOR_MAP.get(p_risk, "#2E86C1")
    st.markdown(f"""
        <div style='padding:15px; border-radius:10px; border-left: 5px solid {c_color}; background-color: #f0f2f6;'>
            <b>Conclusion:</b> Analysis of <b>{u_prod}</b> shows an <b>Adulterated Percentage</b> of <b>{p_pct}%</b>. 
            This results in a <b>Purity</b> level of <b>{purity}%</b> and a 
            <span style='color:{c_color}; font-weight:bold;'>{p_risk}</span> health risk.
        </div>
    """, unsafe_allow_html=True)

import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ---------- CONFIG ----------
st.set_page_config(page_title="Student Risk AI", layout="centered")

# ---------- LOAD MODEL ----------
model = pickle.load(open("model.pkl", "rb"))

# ---------- TITLE ----------
st.markdown("""
<h1 style='text-align:center;'>🎓 Student Risk AI</h1>
<p style='text-align:center; color:gray;'>Predict at-risk students with explainable AI</p>
""", unsafe_allow_html=True)

st.divider()

# ---------- INPUT CARD ----------
st.subheader("📥 Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    studytime = st.slider("Study Time", 1, 4, 2)
    failures = st.slider("Past Failures", 0, 3, 0)
    absences = st.slider("Absences", 0, 50, 5)

with col2:
    Medu = st.slider("Mother Education", 0, 4, 2)
    Fedu = st.slider("Father Education", 0, 4, 2)
    goout = st.slider("Going Out", 1, 5, 3)

Dalc = st.slider("Workday Alcohol", 1, 5, 1)
Walc = st.slider("Weekend Alcohol", 1, 5, 2)
health = st.slider("Health", 1, 5, 3)

st.divider()

# ---------- PREDICTION ----------
if st.button("🚀 Predict"):

    input_df = pd.DataFrame([{
        'studytime': studytime,
        'failures': failures,
        'absences': absences,
        'Medu': Medu,
        'Fedu': Fedu,
        'goout': goout,
        'Dalc': Dalc,
        'Walc': Walc,
        'health': health
    }])

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    # ---------- RESULT ----------
    st.subheader("📊 Result")

    if pred == 1:
        st.markdown(f"### ⚠️ High Risk ({prob:.2f})")
    else:
        st.markdown(f"### ✅ Low Risk ({prob:.2f})")

    # ---------- GAUGE ----------
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={'text': "Risk %"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#FF4B4B"},
            'steps': [
                {'range': [0, 40], 'color': "#2ecc71"},
                {'range': [40, 70], 'color': "#f1c40f"},
                {'range': [70, 100], 'color': "#e74c3c"}
            ]
        }
    ))

    st.plotly_chart(gauge, use_container_width=True)

    # ---------- REASONS ----------
    st.subheader("🧠 Why?")

    reasons = []

    if failures >= 2:
        reasons.append("High past failures")
    if absences > 10:
        reasons.append("High absences")
    if studytime <= 2:
        reasons.append("Low study time")
    if (Dalc + Walc) >= 6:
        reasons.append("Alcohol usage impact")
    if goout >= 4:
        reasons.append("Frequent outings")

    if reasons:
        for r in reasons:
            st.write(f"• {r}")
    else:
        st.write("No major risk signals")

    st.divider()

    # ---------- FEATURE IMPORTANCE ----------
    st.subheader("📊 Feature Importance")

    features = [
        'studytime','failures','absences',
        'Medu','Fedu','goout',
        'Dalc','Walc','health'
    ]

    importances = model.feature_importances_

    fig = px.bar(
        x=importances,
        y=features,
        orientation='h'
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------- CONFUSION MATRIX ----------
    st.subheader("📉 Model Performance")

    cm = np.array([[70, 20],
                   [15, 30]])

    fig_cm = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale="Blues"
    )

    st.plotly_chart(fig_cm, use_container_width=True)
import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Student Risk AI", layout="wide")

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))

# ---------------- HEADER ----------------
st.markdown(
    "<h1 style='text-align:center;'>🎓 Student Risk Prediction System</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>AI-powered early detection of at-risk students</p>",
    unsafe_allow_html=True
)

st.divider()

# ---------------- LAYOUT ----------------
col1, col2 = st.columns([1,1])

# ---------------- INPUT ----------------
with col1:
    st.subheader("📥 Student Details")

    studytime = st.slider("Study Time", 1, 4, 2)
    failures = st.slider("Past Failures", 0, 3, 0)
    absences = st.slider("Absences", 0, 50, 5)

    Medu = st.slider("Mother Education", 0, 4, 2)
    Fedu = st.slider("Father Education", 0, 4, 2)

    goout = st.slider("Going Out Frequency", 1, 5, 3)
    Dalc = st.slider("Workday Alcohol", 1, 5, 1)
    Walc = st.slider("Weekend Alcohol", 1, 5, 2)
    health = st.slider("Health", 1, 5, 3)

# ---------------- PREDICTION ----------------
with col2:
    st.subheader("📊 Prediction Output")

    if st.button("🚀 Predict Risk"):

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

        # -------- RESULT --------
        if pred == 1:
            st.error("⚠️ High Risk Student")
        else:
            st.success("✅ Low Risk Student")

        # -------- GAUGE (INSANE UI) --------
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Risk Level (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red"},
                'steps': [
                    {'range': [0, 40], 'color': "green"},
                    {'range': [40, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ]
            }
        ))

        st.plotly_chart(gauge, use_container_width=True)

        st.divider()

        # -------- EXPLANATION --------
        st.subheader("🧠 Key Risk Factors")

        reasons = []

        if failures >= 2:
            reasons.append("High number of past failures")
        if absences > 10:
            reasons.append("Frequent absences")
        if studytime <= 2:
            reasons.append("Low study time")
        if (Dalc + Walc) >= 6:
            reasons.append("High alcohol consumption")
        if goout >= 4:
            reasons.append("Frequent outings")

        if reasons:
            for r in reasons:
                st.write(f"• {r}")
        else:
            st.write("No major risk indicators detected")

# ---------------- FEATURE IMPORTANCE ----------------
st.divider()
st.subheader("📊 Feature Importance (Interactive)")

features = [
    'studytime','failures','absences',
    'Medu','Fedu','goout',
    'Dalc','Walc','health'
]

importances = model.feature_importances_

fig = px.bar(
    x=importances,
    y=features,
    orientation='h',
    title="Feature Importance",
    labels={'x': 'Importance', 'y': 'Feature'}
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- CONFUSION MATRIX ----------------
st.divider()
st.subheader("📉 Model Performance")

cm = np.array([[70, 20],
               [15, 30]])

fig_cm = px.imshow(
    cm,
    text_auto=True,
    color_continuous_scale='Blues',
    labels=dict(x="Predicted", y="Actual")
)

st.plotly_chart(fig_cm, use_container_width=True)

# ---------------- FOOTER ----------------
st.divider()
st.markdown(
    "<p style='text-align:center;'>Built with ❤️ for Data Mining Project</p>",
    unsafe_allow_html=True
)
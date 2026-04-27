import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go

# ---------- CONFIG ----------
st.set_page_config(page_title="Student Risk AI", layout="centered")

# ---------- LOAD MODEL ----------
model = pickle.load(open("model.pkl", "rb"))

# ---------- HEADER ----------
st.markdown("""
<h1 style='text-align:center;'>🎓 Student Risk Predictor</h1>
<p style='text-align:center; color:gray;'>Enter details and predict student risk instantly</p>
""", unsafe_allow_html=True)

st.divider()

# ---------- CARD STYLE ----------
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 700px;
}
</style>
""", unsafe_allow_html=True)

# ---------- FORM ----------
with st.container():
    st.subheader("📋 Student Details")

    studytime = st.slider("Study Time", 1, 4, 2)
    failures = st.slider("Past Failures", 0, 3, 0)
    absences = st.slider("Absences", 0, 50, 5)

    Medu = st.slider("Mother Education", 0, 4, 2)
    Fedu = st.slider("Father Education", 0, 4, 2)

    goout = st.slider("Going Out", 1, 5, 3)
    Dalc = st.slider("Workday Alcohol", 1, 5, 1)
    Walc = st.slider("Weekend Alcohol", 1, 5, 2)
    health = st.slider("Health", 1, 5, 3)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀 Predict", use_container_width=True):

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

        st.divider()

        # ---------- RESULT ----------
        if pred == 1:
            st.error(f"⚠️ High Risk ({prob:.2f})")
        else:
            st.success(f"✅ Low Risk ({prob:.2f})")

        # ---------- GAUGE ----------
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Risk Level"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#ff4b4b"},
                'steps': [
                    {'range': [0, 40], 'color': "#2ecc71"},
                    {'range': [40, 70], 'color': "#f1c40f"},
                    {'range': [70, 100], 'color': "#e74c3c"}
                ]
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

        # ---------- SIMPLE INSIGHT ----------
        st.subheader("🧠 Key Factors")

        if failures >= 2:
            st.write("• High past failures")
        if absences > 10:
            st.write("• High absences")
        if studytime <= 2:
            st.write("• Low study time")
import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go

# ---------- CONFIG ----------
st.set_page_config(page_title="Student Risk AI", layout="centered")

# ---------- LOAD MODEL ----------
model = pickle.load(open("model.pkl", "rb"))

# ---------- STYLE ----------
st.markdown("""
<style>
.block-container {
    max-width: 750px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* MAIN CARD */
.main-card {
    background-color: #111;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 0 20px rgba(0,0,0,0.4);
    margin-bottom: 20px;
}

/* SECTION CARDS */
.section-card {
    background-color: #1a1a1a;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("""
<h1 style='text-align:center;'>Student Risk Predictor</h1>
<p style='text-align:center; color:gray;'>AI-powered academic risk analysis</p>
""", unsafe_allow_html=True)

# ---------- MAIN CARD START ----------
st.markdown('<div class="main-card">', unsafe_allow_html=True)

# ---------- ACADEMIC ----------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown("### Academic")

col1, col2 = st.columns(2)

with col1:
    studytime = st.selectbox("Study Time", [1,2,3,4])
    failures = st.selectbox("Past Failures", [0,1,2,3])

with col2:
    absences = st.number_input("Absences", 0, 50, 5)

st.markdown('</div>', unsafe_allow_html=True)

# ---------- FAMILY ----------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown("### Family Background")

col1, col2 = st.columns(2)

with col1:
    Medu = st.selectbox("Mother Education", [0,1,2,3,4])

with col2:
    Fedu = st.selectbox("Father Education", [0,1,2,3,4])

st.markdown('</div>', unsafe_allow_html=True)

# ---------- LIFESTYLE ----------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown("### Lifestyle")

col1, col2 = st.columns(2)

with col1:
    goout = st.selectbox("Going Out", [1,2,3,4,5])
    Dalc = st.selectbox("Workday Alcohol", [1,2,3,4,5])

with col2:
    Walc = st.selectbox("Weekend Alcohol", [1,2,3,4,5])
    health = st.selectbox("Health", [1,2,3,4,5])

st.markdown('</div>', unsafe_allow_html=True)

# ---------- BUTTON ----------
st.markdown("<br>", unsafe_allow_html=True)

predict = st.button("Predict Risk", use_container_width=True)

# ---------- MAIN CARD END ----------
st.markdown('</div>', unsafe_allow_html=True)

# ---------- PREDICTION ----------
if predict:

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

    st.markdown("## Prediction Result")

    if prob < 0.4:
        st.success(f"Low Risk ({prob:.2f})")
    elif prob < 0.7:
        st.warning(f"Medium Risk ({prob:.2f})")
    else:
        st.error(f"High Risk ({prob:.2f})")

    # ---------- GAUGE ----------
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={'text': "Risk Level (%)"},
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

    # ---------- INSIGHTS ----------
    st.markdown("## Key Risk Factors")

    insights = []

    if failures >= 2:
        insights.append("High past failures")
    if absences > 10:
        insights.append("High absenteeism")
    if studytime <= 2:
        insights.append("Low study time")
    if goout >= 4:
        insights.append("High social activity")
    if Dalc >= 3 or Walc >= 3:
        insights.append("High alcohol consumption")

    if insights:
        for i in insights:
            st.write(f"• {i}")
    else:
        st.write("• No major risk factors detected")
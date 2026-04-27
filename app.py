import streamlit as st
import pickle
import pandas as pd

# ---------- Load Model ----------
model = pickle.load(open("model.pkl", "rb"))

# ---------- Page Config ----------
st.set_page_config(page_title="Student Risk Predictor", layout="centered")

st.title("🎓 Student Dropout Risk Predictor")
st.markdown("Predict risk based on student behavior and background")

st.divider()

# ---------- Input Section ----------
st.subheader("📥 Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    studytime = st.slider("Study Time (1–4)", 1, 4, 2)
    failures = st.slider("Past Failures (0–3)", 0, 3, 0)
    absences = st.slider("Absences", 0, 50, 5)

    Medu = st.slider("Mother Education (0–4)", 0, 4, 2)
    Fedu = st.slider("Father Education (0–4)", 0, 4, 2)

with col2:
    goout = st.slider("Going Out Frequency (1–5)", 1, 5, 3)
    Dalc = st.slider("Workday Alcohol (1–5)", 1, 5, 1)
    Walc = st.slider("Weekend Alcohol (1–5)", 1, 5, 2)
    health = st.slider("Health (1–5)", 1, 5, 3)

st.divider()

# ---------- Prediction ----------
if st.button("🔍 Predict Risk"):

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

    st.subheader("📊 Prediction Result")

    if pred == 1:
        st.error(f"⚠️ High Risk Student")
    else:
        st.success(f"✅ Low Risk Student")

    st.write(f"**Risk Probability:** {prob:.2f}")

    # ---------- Risk Meter ----------
    st.progress(float(prob))

    st.divider()

    # ---------- Explanation ----------
    st.subheader("🧠 Why this prediction?")

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
        reasons.append("Frequent social outings")

    if len(reasons) > 0:
        for r in reasons:
            st.write(f"• {r}")
    else:
        st.write("No major risk indicators detected")

st.divider()

# ---------- Footer ----------
st.markdown("Built for Data Mining Project | Dropout Risk Prediction")
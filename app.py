# import streamlit as st
# import pickle
# import pandas as pd
# import plotly.graph_objects as go

# # ---------- CONFIG ----------
# st.set_page_config(page_title="Student Risk AI", layout="centered")

# # ---------- LOAD MODEL ----------
# @st.cache_resource
# def load_model():
#     return pickle.load(open("model.pkl", "rb"))

# model = load_model()

# # ---------- HEADER ----------
# st.title("Student Risk Predictor")
# st.write("Enter student details and get risk prediction instantly.")

# # ---------- FORM ----------
# with st.form("student_form"):

#     st.subheader("Academic")

#     col1, col2 = st.columns(2)

#     with col1:
#         studytime = st.selectbox("Study Time", [1,2,3,4])
#         failures = st.selectbox("Past Failures", [0,1,2,3])

#     with col2:
#         absences = st.number_input("Absences", 0, 50, 5)

#     st.subheader("Family Background")

#     col1, col2 = st.columns(2)

#     with col1:
#         Medu = st.selectbox("Mother Education", [0,1,2,3,4])

#     with col2:
#         Fedu = st.selectbox("Father Education", [0,1,2,3,4])

#     st.subheader("Lifestyle")

#     col1, col2 = st.columns(2)

#     with col1:
#         goout = st.selectbox("Going Out", [1,2,3,4,5])
#         Dalc = st.selectbox("Workday Alcohol", [1,2,3,4,5])

#     with col2:
#         Walc = st.selectbox("Weekend Alcohol", [1,2,3,4,5])
#         health = st.selectbox("Health", [1,2,3,4,5])

#     submit = st.form_submit_button("Predict")

# # ---------- PREDICTION ----------
# if submit:

#     input_df = pd.DataFrame([{
#         'studytime': studytime,
#         'failures': failures,
#         'absences': absences,
#         'Medu': Medu,
#         'Fedu': Fedu,
#         'goout': goout,
#         'Dalc': Dalc,
#         'Walc': Walc,
#         'health': health
#     }])

#     pred = model.predict(input_df)[0]
#     prob = model.predict_proba(input_df)[0][1]

#     st.markdown("### Result")

#     if prob < 0.4:
#         st.success(f"Low Risk ({prob:.2f})")
#     elif prob < 0.7:
#         st.warning(f"Medium Risk ({prob:.2f})")
#     else:
#         st.error(f"High Risk ({prob:.2f})")

#     # ---------- GAUGE ----------
#     fig = go.Figure(go.Indicator(
#         mode="gauge+number",
#         value=prob * 100,
#         title={'text': "Risk Level (%)"},
#         gauge={
#             'axis': {'range': [0, 100]},
#             'bar': {'color': "#ff4b4b"},
#             'steps': [
#                 {'range': [0, 40], 'color': "#2ecc71"},
#                 {'range': [40, 70], 'color': "#f1c40f"},
#                 {'range': [70, 100], 'color': "#e74c3c"}
#             ]
#         }
#     ))

#     st.plotly_chart(fig, use_container_width=True)

#     # ---------- INSIGHTS ----------
#     st.markdown("### Key Risk Factors")

#     insights = []

#     if failures >= 2:
#         insights.append("High past failures")
#     if absences > 10:
#         insights.append("High absenteeism")
#     if studytime <= 2:
#         insights.append("Low study time")
#     if goout >= 4:
#         insights.append("High social activity")
#     if Dalc >= 3 or Walc >= 3:
#         insights.append("High alcohol consumption")

#     if insights:
#         for i in insights:
#             st.write(f"- {i}")
#     else:
#         st.write("- No major risk factors detected")


import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go

# ---------- CONFIG ----------
st.set_page_config(page_title="Student Risk AI", layout="centered")

features = [
    'Curricular units 2nd sem (approved)',
    'Curricular units 1st sem (approved)',
    'Curricular units 2nd sem (grade)',
    'Curricular units 1st sem (grade)',
    'Tuition fees up to date',
    'Admission grade',
    'Previous qualification (grade)',
    'Age at enrollment',
    'Course',
    'Curricular units 2nd sem (evaluations)',
    'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (enrolled)'
]

# ---------- LOAD ----------
@st.cache_resource
def load_all():
    model = pickle.load(open("model.pkl","rb"))
    scaler = pickle.load(open("scaler.pkl","rb"))
    features = pickle.load(open("features.pkl","rb"))
    return model, scaler, features

model, scaler, features = load_all()

# ---------- HEADER ----------
st.title("Student Risk AI")
st.write("Predict student dropout risk using academic and financial data")

# ---------- FORM ----------
with st.form("student_form"):

    st.subheader("Academic Performance")

    col1, col2 = st.columns(2)

    with col1:
        approved1 = st.number_input("1st Sem Approved Subjects", 0, 15, 5)
        grade1 = st.number_input("1st Sem Grade", 0.0, 20.0, 10.0)
        eval1 = st.number_input("1st Sem Evaluations", 0, 20, 6)
        enrolled1 = st.number_input("1st Sem Enrolled Subjects", 0, 15, 6)

    with col2:
        approved2 = st.number_input("2nd Sem Approved Subjects", 0, 15, 5)
        grade2 = st.number_input("2nd Sem Grade", 0.0, 20.0, 10.0)
        eval2 = st.number_input("2nd Sem Evaluations", 0, 20, 6)

    st.subheader("Financial & Background")

    col1, col2 = st.columns(2)

    with col1:
        fees = st.selectbox("Tuition Fees Up To Date", [0,1])
        admission = st.number_input("Admission Grade", 0.0, 200.0, 120.0)
        prev_grade = st.number_input("Previous Qualification Grade", 0.0, 200.0, 120.0)

    with col2:
        age = st.number_input("Age at Enrollment", 15, 80, 20)
        course = st.number_input("Course Code", 1, 20, 1)

    submit = st.form_submit_button("Predict Risk")

# ---------- PREDICTION ----------
if submit:

    input_data = pd.DataFrame([{
        'Curricular units 2nd sem (approved)': approved2,
        'Curricular units 1st sem (approved)': approved1,
        'Curricular units 2nd sem (grade)': grade2,
        'Curricular units 1st sem (grade)': grade1,
        'Tuition fees up to date': fees,
        'Admission grade': admission,
        'Previous qualification (grade)': prev_grade,
        'Age at enrollment': age,
        'Course': course,
        'Curricular units 2nd sem (evaluations)': eval2,
        'Curricular units 1st sem (evaluations)': eval1,
        'Curricular units 1st sem (enrolled)': enrolled1
    }])

    input_data = input_data[features]

    # scale
    scaled = scaler.transform(input_data)

    prob = model.predict_proba(scaled)[0][1]
    pred = model.predict(scaled)[0]

    st.markdown("### Result")

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
        title={'text': "Dropout Risk (%)"},
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
    st.markdown("### Key Risk Factors")

    insights = []

    if approved1 < 3 or approved2 < 3:
        insights.append("Low number of approved subjects")

    if grade1 < 10 or grade2 < 10:
        insights.append("Low academic performance")

    if fees == 0:
        insights.append("Tuition fees not paid")

    if eval1 < 3 or eval2 < 3:
        insights.append("Low exam participation")

    if insights:
        for i in insights:
            st.write(f"- {i}")
    else:
        st.write("- No major risk factors detected")
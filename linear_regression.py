import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Study Hours Predictor")

st.title("🎓 Study Hours Prediction App")

@st.cache_data
def load_data():
    df = pd.read_excel("student_study_survey_500.xlsx")
    df = df.dropna().astype(float)
    return df

df = load_data()

X = df[["Age","Year_of_Study","CGPA","Last_Exam_Percentage",
        "Sleep_Hours","Screen_Time_Hours","Attendance",
        "Commute_Time_Min","Motivation_Level"]]

y = df["Study_Hours"]

@st.cache_resource
def train_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model(X, y)

st.sidebar.header("Enter Student Details")

age = st.sidebar.number_input("Age", 15, 30)
year = st.sidebar.number_input("Year of Study", 1, 5)
cgpa = st.sidebar.number_input("CGPA", 0.0, 10.0)
last_exam = st.sidebar.number_input("Last Exam %", 0.0, 100.0)
sleep = st.sidebar.number_input("Sleep Hours", 0.0, 12.0)
screen = st.sidebar.number_input("Screen Time", 0.0, 12.0)
attendance = st.sidebar.number_input("Attendance %", 0.0, 100.0)
commute = st.sidebar.number_input("Commute Time (min)", 0, 120)
motivation = st.sidebar.number_input("Motivation Level (1-5)", 1, 5)

if st.button("Predict Study Hours"):
    input_data = pd.DataFrame([[age, year, cgpa, last_exam,
                                sleep, screen, attendance,
                                commute, motivation]],
        columns=X.columns)

    prediction = model.predict(input_data)[0]
    st.success(f"📚 Predicted Study Hours: {round(prediction,2)} hours")

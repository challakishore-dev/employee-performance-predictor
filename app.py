
import streamlit as st, pandas as pd, joblib, os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Employee Performance Predictor", layout="wide")
st.title("Employee Performance Predictor ")

@st.cache_resource
def load_model():
    return joblib.load("models/model.pkl")

if not os.path.exists("models/model.pkl"):
    st.warning("Run python main.py first")
    st.stop()

model=load_model()

col1,col2=st.columns(2)
with col1:
    age=st.slider("Age",22,55,30)
    experience=st.slider("Experience",1,20,5)
    salary=st.slider("Salary",30000,150000,60000,1000)
    training_hours=st.slider("Training Hours",5,100,40)
with col2:
    projects=st.slider("Projects",1,10,5)
    attendance=st.slider("Attendance",70,100,92)
    department=st.selectbox("Department",["IT","HR","Sales","Finance"])

row=pd.DataFrame([{
    "age":age,"experience":experience,"salary":salary,
    "training_hours":training_hours,"projects":projects,
    "attendance":attendance,"department":department
}])

if st.button("Predict Performance"):
    pred=model.predict(row)[0]
    st.success(f"Predicted Performance: {pred}")

st.subheader("Saved Project Outputs")
for img in ["outputs/confusion_matrix.png","outputs/feature_importance.png"]:
    if os.path.exists(img):
        st.image(img)

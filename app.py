import pickle
import numpy as np
import streamlit as st

with open("diabetes.pkl", 'rb') as f:
    model = pickle.load(f)
    
st.title("Diabetes prediction app")
st.write("Enter the details to chek person have diabetes or not")

pregnancies=st.number_input("Pregnancies",0,20,1)
glucose=st.number_input("Glucose level",50,300,100)
blood_pressuer=st.number_input("Blood pressure",30,150,70)
skin_thickness=st.number_input("Skin thikness",0,100,20)
insulin=st.number_input("Insulin level",0,900,80)
bmi=st.number_input("BMI",10.0,70.0,25.0)
dpf=st.number_input("Diabetes pedigree",0.0,3.0,0.3)
age=st.number_input("Age",10,100,30)

if st.button("Predict"):
    input_data=np.array([[pregnancies, glucose, blood_pressuer, skin_thickness, insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.error("The person is likely to have diabetes")
    else:
        st.success("The person is not likely to have diabetes")
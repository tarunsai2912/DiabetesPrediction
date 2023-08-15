# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 12:49:24 2023

@author: tarun
"""
import pandas as pd
import numpy as np 
import pickle #used for importing the model
import streamlit as st #used for deploying the model


#loading the saved model
#copy the path of file and paste
#replace backslash with forwardslash
pickled_model=pickle.load(open('C:/Users/tarun/Downloads/machine leaarning deploy 1/regmodel1.pkl','rb'))


#creating a function for prediction
def diabetes_prediction(input_data):
    

    #change the input data to numpy data
    input_numpy_data = np.asarray(input_data)

    #Reshape the input data as it is single input data
    input_reshaped_data = input_numpy_data.reshape(1,-1)

    prediction = pickled_model.predict(input_reshaped_data)

    if prediction[0]==0:
        return("Non-Diabetic")
    else:
        return("Diabetic")
    
    
    
def main():
    
    #giving a title
    st.title("Diabetes Risk Prediction Web App")
    st.write("Kindly enter the following data")
    st.image("https://www.howard-finley.co.uk/wp-content/uploads/2020/06/dreamstime_l_180162487-scaled.jpg")
    
    #getting the input data from user
    
    Pregnancies = st.text_input("Enter No Of Pregnancies")
    Glucose = st.text_input("Enter Glucose Level")
    BloodPressure = st.text_input("Enter Blood Pressure Level")
    SkinThickness = st.text_input("Enter Skin Thickness Level")
    Insulin = st.text_input("Enter Insulin Level")
    BMI = st.text_input("Enter BMI Value")
    DiabetesPedigreeFunction = st.text_input("Enter Diabetes Pedigree Function Value")
    Age = st.text_input("Enter Age")
    
    #code for prediction
    #should be empty to store the result
    diagnosis = ''
    
    #creating a buttton for prediction
    if st.button("Diabetes Test Result"):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    
    st.success(diagnosis)
    
    st.image("https://th.bing.com/th/id/R.c5242212f640b9576ba2a24e3f417c16?rik=8assoFITcfDOuA&riu=http%3a%2f%2fcannabidiol360.com%2fwp-content%2fuploads%2f2018%2f02%2fsymptoms-of-diabetes-1.jpg&ehk=J9gjoDFp0IfqqH%2b5qTUk%2f7hhjNB1IaHvIBpiN%2f%2fB5vs%3d&risl=1&pid=ImgRaw&r=0")
    
    
    
#we use main func inorder to perform when it id=s directly called by commd prompt
if __name__ == "__main__":
    main() 
    
     
    

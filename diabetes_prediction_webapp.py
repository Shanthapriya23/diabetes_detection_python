# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 22:44:35 2024

@author: PRIYA
"""
import numpy as np
import pickle
import streamlit as st

#loading the saved model
loaded_model = pickle.load(open('trained_model.sav','rb'))

def diabetes_prediction(input_data):
    #convert data to numpy array:
    data_np_arr = np.asarray(input_data)

    #reshape the array as we are predicting for one instance :
    reshaped_input_data = data_np_arr.reshape(1,-1)

    prediction = loaded_model.predict(reshaped_input_data)
    print(prediction)
    if prediction[0]==0:
      return 'patient is not diabetic'
    else:
      return 'patient is diabetic'

def main():
    #set title of the web app
    st.title('Diabetes Prediction Web App')
    #create fields to enter user input values
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the person')
    
    #code for prediction
    diagnosis = ' '
    
    #create a button to submit the input data and get prediction results
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    st.success(diagnosis)   
        
if __name__ == '__main__' :
    main()     
        
        

import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('lgb.pkl')

st.title('Demo model of insurance cost prediction')

# get params
gender = st.selectbox('Who you are? ', options=['Male', 'Female'])
age = st.slider('Specify your age: ', 18, 65, 25)
height = st.slider('Specify your height: ', 120, 220, 175)
weight = st.slider('Specify your weight', 30, 150, 65)
is_smoker = st.selectbox('Are you smoking? ', options=['No', 'Yes'])
children = st.selectbox('How many children do you have? ', ['0', '1', '2', '3', '4', '5 or more'])

# separate output part
st.write(' ')
st.write(' ')
st.write('Press here to get an estimated cost')

if st.button('Predict'):
    smoker = 1 if is_smoker == 'Yes' else 0
    if children == '5 or more':
        children = '5'
    gender = 1 if gender == 'Male' else 0
    bmi = weight / (height / 100) ** 2
    # build vector to predict
    x = np.array([gender, gender, bmi, int(children), smoker]).reshape([1, 5])
    # get a prediction
    y = model.predict(x)[0]
    # round answer
    y = np.round(y, 2)
    st.write('The expected cost is: ')
    st.write(y)

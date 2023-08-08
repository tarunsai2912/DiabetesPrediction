# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

"""
import pandas as pd
import numpy as np
import pickle



#loading the saved model
pickled_model=pickle.load(open('C:/Users/tarun/Downloads/machine leaarning deploy 1/regmodel1.pkl','rb'))



#making a predictive system
input_data = (5,116,74,0,0,25.6,0.201,30)

#change the input data to numpy data
input_numpy_data = np.asarray(input_data)

#Reshape the input data as it is single input data
input_reshaped_data = input_numpy_data.reshape(1,-1)

prediction = pickled_model.predict(input_reshaped_data)

if prediction[0]==0:
    print("Non-Diabetic")
else:
    print("Diabetic")
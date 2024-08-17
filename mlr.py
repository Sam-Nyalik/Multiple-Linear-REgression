#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 21:54:16 2024

@author: sam-nyalik
"""

#Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Import dataset
data_set = pd.read_csv('/home/sam-nyalik/Desktop/ML/50_Startups.csv')
print(data_set)

#Extract the indepenedent and dependent variables
#independent
x = data_set.iloc[:, :-1]
y = data_set.iloc[:, 4]

#Encoding dummy variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_x = LabelEncoder()
data_set.iloc[:, 3] = labelencoder_x.fit_transform(data_set.iloc[:, 3])
column_transformer = ColumnTransformer(
    transformers= [
        ('onehot', OneHotEncoder(), [3])
        ],
    remainder='passthrough'
    )
x = column_transformer.fit_transform(x)
x = x[:, 1:]

#Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#Fitting the MLR model to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Predicting the test set result
y_pred = regressor.predict(x_test)


#Check the score for the training dataset and test dataset
print("Training Score: ", regressor.score(x_train, y_train))
print("Test Score: ",regressor.score( x_test, y_test))

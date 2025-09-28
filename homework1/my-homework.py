import pandas as pd
import numpy as np
print(pd.__version__)

df=pd.read_csv("D:\python\MLZoomcamp\machine-learning-zoomcamp-homework\homework1\car_fuel_efficiency.csv")
print(df.head())
print(df.describe)
print("nunique:",df.fuel_type.nunique())#Fuel types 
print(df.isnull().sum())#Missing values
print(df["fuel_efficiency_mpg"].max())#Max fuel efficiency?
print('-------')
print(df.groupby('model_year')['horsepower'].median())#Median value of horsepower 
print("Sum of weights",df['vehicle_weight'].sum())#?选项没有
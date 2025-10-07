import pandas as pd
import numpy as np
print(pd.__version__)

df=pd.read_csv("D:\python\MLZoomcamp\machine-learning-zoomcamp-homework\homework1\car_fuel_efficiency.csv")
'''
print(df.head())
print(df.describe)
print("nunique:",df.fuel_type.nunique())#Fuel types 
print(df.isnull().sum())#Missing values

#Q6. Median value of horsepower
print("df['horsepower'].median()",df['horsepower'].median())
most_freq=df['horsepower'].mode()[0]
print("most_freq",most_freq)
df_miss=df['horsepower'].isnull()
print('-------')
print(df['horsepower'].head(15))
df['horsepower'].fillna(value=most_freq, inplace=True)
print('+++')
print(df['horsepower'].head(15))
print("df['horsepower'].median()",df['horsepower'].median())
'''
#Q7. Sum of weights
df_Asia=df[df['origin'] == 'Asia']#Select all the cars from Asia 
df_Asia_data=df_Asia[['vehicle_weight','model_year']].head(7)#Select only columns vehicle_weight and model_year
X=df_Asia_data.to_numpy()
XTX=np.dot(X,X.T)
print(X.shape)
XTX_Inv=np.linalg.inv(XTX)
y=np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
w=
print(y)
#X=df.head(7).to_numpy()
#XTX=np.dot(X,X.T)
#print(XTX)
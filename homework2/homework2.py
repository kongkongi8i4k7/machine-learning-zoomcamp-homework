import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'D:\python\MLZoomcamp\machine-learning-zoomcamp-homework\homework2\car_fuel_efficiency.csv')


#
df=df[['engine_displacement','horsepower','vehicle_weight','model_year','fuel_efficiency_mpg']]
base = ['engine_displacement','horsepower','vehicle_weight','model_year']  # 不要包含 fuel_efficiency_mpg   
#Question 1. Missing values
#cf.isnull().sum()  
#horsepower             708 

#Question 2. Median for horse power
df['horsepower'].median()#np.float64(149.0)

#Question 3. Filling NAs

n = len(df)
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test
#print(n_val, n_test, n_train)#1940 1940 5824

df_train = df[:n_train]
df_val = df[n_train:n_train + n_val]
df_test = df[n_train + n_val:]

idx = np.arange(n)
np.random.seed(42)
np.random.shuffle(idx)

df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train:n_train+n_val]]
df_test = df.iloc[idx[n_train+n_val:]]
#df_train.head(5)

print(len(df_train))#5824
print(len(df_val))#1940
print(len(df_test))#1940

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
y_train = np.log1p(df_train.fuel_efficiency_mpg.values)
y_val = np.log1p(df_val.fuel_efficiency_mpg.values)
y_test = np.log1p(df_test.fuel_efficiency_mpg.values)

#print(df_train.head(5))


def prepare_X(df):
    df_num = df[base]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X

def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)
    
    return w_full[0], w_full[1:]

def rmse(y_pred, y_train):
    se = (y_pred - y_train) **2
    means = se.mean()
    return np.sqrt(means)   


#fill it with 0 
df_0_train = df_train[base].fillna(0).values
w0_0, w_0 = train_linear_regression(df_0_train, y_train)
y_pred_0 = w0_0 + df_0_train.dot(w_0)
rmse_0 = rmse(y_pred_0, y_train) 
rmse_0 = round(rmse_0, 2)#np.float64(0.04)


#fill it with mean 
horsepower_mean = df['horsepower'].mean()
df_mean_train = df_train[base].fillna(horsepower_mean).values
w0_mean, w_mean = train_linear_regression(df_mean_train, y_train)
y_pred_mean = w0_mean + df_mean_train.dot(w_mean)
rmse_mean = rmse(y_pred_mean, y_train)#
rmse_mean = round(rmse_mean, 2)#np.float64(0.04)


#Question 4



def train_linear_regression_reg(X, y, r):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])

    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)
    
    return w_full[0], w_full[1:]


for r in [0, 0.01, 0.1, 1, 5, 10, 100]:
    X_train = prepare_X(df_train[base])
    w0, w = train_linear_regression_reg(X_train, y_train, r=r)

    X_val = prepare_X(df_val)
    y_pred = w0 + X_val.dot(w)
    score = rmse(y_val, y_pred)
    score1 = round(score, 2)
    print(r, w0, score)

#Question 5


def seed_split(i):
    n = len(df)
    n_val = int(n * 0.2)
    n_test = int(n * 0.2)
    n_train = n - n_val - n_test

    idx = np.arange(n)
    np.random.seed(i)
    np.random.shuffle(idx)

    df_train = df.iloc[idx[:n_train]].reset_index(drop=True)
    df_val = df.iloc[idx[n_train:n_train+n_val]].reset_index(drop=True)
    df_test = df.iloc[idx[n_train+n_val:]].reset_index(drop=True)

    y_train = np.log1p(df_train.fuel_efficiency_mpg.values)
    y_val = np.log1p(df_val.fuel_efficiency_mpg.values)
    y_test = np.log1p(df_test.fuel_efficiency_mpg.values)

    return df_train, df_val, df_test, y_train, y_val, y_test

'''


r = 0.01   
scores = []
for i in range(10):
    print('-----------------')
    print(f"seed = {i}")

    df_train, df_val, df_test, y_train, y_val, y_test = seed_split(i)
    X_train = prepare_X(df_train)
    w0, w = train_linear_regression_reg(X_train, y_train, r=r)

    X_val = prepare_X(df_val)
    y_pred = w0 + X_val.dot(w)
    score = rmse(y_val, y_pred)
    scores.append(score)
    print(f"r = {r}, w0 = {w0:.4f}, RMSE = {score:f}")

std_score = np.std(scores)
print(std_score,round(std_score, 3))
'''
#Question 6. Evaluation on test

df_train, df_val, df_test, y_train, y_val, y_test = seed_split(9)
df_full_train = pd.concat([df_train, df_val])
df_full_train = df_full_train.reset_index(drop=True)
X_full_train = prepare_X(df_full_train)
y_full_train = np.concatenate([y_train, y_val])
w0, w = train_linear_regression_reg(X_full_train, y_full_train, r=0.001)

X_test = prepare_X(df_test)
y_pred = w0 + X_test.dot(w)
score = rmse(y_test, y_pred)
score#np.float64(0.03919613644482336)


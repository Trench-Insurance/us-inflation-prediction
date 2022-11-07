import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error
import tensorflow as tf
from statsmodels.tsa.ar_model import AutoReg

PREDICTION_WIDTH = 24

# df = pd.read_csv("dataset/CPIAUCSL.csv")
df = pd.read_csv("dataset/CPIAUCNS.csv")

scaler = StandardScaler()

df["CPIAUCNS"] = scaler.fit_transform(df["CPIAUCNS"].values.reshape(-1,1))

print(df.head())

x = df.index.values.reshape(-1,1)
y = df["CPIAUCNS"].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=False)

print(x_train.shape, x_test.shape)

model = AutoReg(y_train, lags=10)
model.fit()

y_pred = model.predict(len(y_train), len(y_train) + len(y_test) - 1)

rmse =  mean_squared_error(y_test, y_pred, squared=True)
print(f"auto_regression rmse: {rmse}")

# auto_regression rmse: 0.06099068252540488

x = model.predict(
    len(y_train), len(y_train) + PREDICTION_WIDTH - 1
)
x = np.array(x)
x = scaler.inverse_transform(x.reshape(-1,1))

print(x)




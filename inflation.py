import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
import functions as functions

LABEL_WIDTH = 1
WINDOW_SIZE = 2

# df = pd.read_csv("dataset/CPIAUCSL.csv")
df = pd.read_csv("dataset/CPIAUCNS.csv")
# df.dropna(inplace=True)

# df["Index"] = normalize(df["Index"].values.reshape(-1,1), axis=0)
scaler = StandardScaler()

df["CPIAUCNS"] = scaler.fit_transform(df["CPIAUCNS"].values.reshape(-1,1))

print(df.head())

x = df.index.values.reshape(-1,1)
y = df["CPIAUCNS"].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=False)

print(x_train.shape, x_test.shape)

# functions.linear_regression(x_train, y_train, x_test, y_test)
auto_reg_model = functions.auto_regression(y_train, y_test)




# y_pred = scaler.inverse_transform(y_pred.reshape(-1,1))
# y_test = scaler.inverse_transform(y_test.reshape(-1,1))

# print(list(zip(y_pred, y_test)))

# nn_dense_model = functions.nn_dense()
# functions.train(nn_dense_model ,x_train, y_train, epochs=50, batch_size=1)
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import functions as functions

LABEL_WIDTH = 1
WINDOW_SIZE = 2

df = pd.read_csv("dataset/cpi-us_zip/data/cpiai_csv.csv")
df["Month"] = df.index
# df.dropna(inplace=True)

df["Index"] = normalize(df["Index"].values.reshape(-1,1), axis=0)

print(df.head())

x = df["Month"].values.reshape(-1,1)
y = df["Index"].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

print(x_train.shape, x_test.shape)

functions.linear_regression(x_train, y_train, x_test, y_test)
functions.auto_regression(y_train, y_test)

# nn_dense_model = functions.nn_dense()
# functions.train(nn_dense_model ,x_train, y_train, epochs=50, batch_size=1)
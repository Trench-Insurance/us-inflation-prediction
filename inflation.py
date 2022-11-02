import pandas as pd
import numpy as np
import tensorflow as tf

import functions as functions

LABEL_WIDTH = 1
WINDOW_SIZE = 2

df = pd.read_csv("dataset/cpi-us_zip/data/cpiai_csv.csv")
df["Month"] = df.index
df.dropna(inplace=True)

print(df.head())

x = df["Month"].values.reshape(-1,1)
y = df["Inflation"].values

functions.linear_regression(x,y)

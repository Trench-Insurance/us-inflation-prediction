import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


LABEL_WIDTH = 1
WINDOW_SIZE = 2

df = pd.read_csv("dataset/cpi-us_zip/data/cpiai_csv.csv")
# df[["Year", "Month", "Day"]] = df["Date"].str.split("-", 2, expand=True)
df["Month"] = df.index
df.dropna(inplace=True)

print(df.head())
# df.plot("Month",["Index","Inflation"])
# plt.show()
# plt.savefig("fig")

x = df["Month"].values.reshape(-1,1)
y = df["Inflation"].values

# Train the model
model = LinearRegression()
model.fit(x, y)

y_pred = pd.Series(model.predict(x), index=df.index)

print(y_pred)
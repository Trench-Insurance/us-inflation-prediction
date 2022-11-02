import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def linear_regression(x,y):
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)

    rmse =  mean_squared_error(y, y_pred, squared=True)
    print(rmse)
    # 0.43679908465250444


def plot_index_inflation(df):
    df.plot("Month",["Index","Inflation"])
    plt.show()
    plt.savefig("fig")
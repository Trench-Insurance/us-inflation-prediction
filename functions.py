import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from statsmodels.tsa.ar_model import AutoReg

def plot_index_inflation(df):
    df.plot("Month",["Index","Inflation"])
    plt.show()
    plt.savefig("fig")


def linear_regression(x_train, y_train, x_test, y_test):
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    rmse =  mean_squared_error(y_test, y_pred, squared=True)
    print(f"linear_regression rmse: {rmse}")

    # linear_regression rmse: 7.164442386810773e-05


def auto_regression(y_train, y_test):
    model = AutoReg(y_train, lags=10)
    model_fit = model.fit()

    y_pred = model_fit.predict(len(y_train), len(y_train) + len(y_test) - 1)

    rmse =  mean_squared_error(y_test, y_pred, squared=True)
    print(f"auto_regression rmse: {rmse}")
    
    # auto_regression rmse: 0.0003492280629135441


def nn_dense():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=1)
    ])
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")])

    return model

    # Epoch 50/50
    # 1031/1031 [==============================] - 1s 1ms/step - loss: 4.3914e-04 - rmse: 0.0210


def train(model, x_train, y_train, epochs=50, batch_size=32):
    model.build(x_train.shape)
    print(model.summary())
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, Normalizer
import tensorflow as tf

train_df = pd.read_csv(
    "dataset/insurance-customer/train.csv")
test_df = pd.read_csv("dataset/insurance-customer/test.csv")

train_df.drop(["id","Count_3-6_months_late","Count_6-12_months_late","Count_more_than_12_months_late"], axis=1, inplace=True)
test_df.drop(["id","Count_3-6_months_late","Count_6-12_months_late","Count_more_than_12_months_late"], axis=1, inplace=True)

train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

train_df["residence_area_type"] = LabelEncoder().fit_transform(train_df["residence_area_type"])
train_df["sourcing_channel"] = LabelEncoder().fit_transform(train_df["sourcing_channel"])
test_df["residence_area_type"] = LabelEncoder().fit_transform(test_df["residence_area_type"])
test_df["sourcing_channel"] = LabelEncoder().fit_transform(test_df["sourcing_channel"])

train = np.array(train_df.values)
test = np.array(test_df.values)

train = Normalizer().fit_transform(train)
test = Normalizer().fit_transform(test)

x_train = train[:, :-1]
y_train = train[:, -1]
x_test = test[:, :-1]
y_test = test[:, -1]

print(x_train.shape)
print(y_train.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])


model.compile(optimizer=tf.keras.optimizers.Adam(lr = 0.01),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=["accuracy"])

model.fit(x_train, y_train, epochs=10)
# model.evaluate(x_test,  y_test, verbose=2)

# not learning

# random forest?
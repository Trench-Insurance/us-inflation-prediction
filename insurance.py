import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, Normalizer
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

train_df = pd.read_csv(
    "dataset/insurance-customer/train.csv")
test_df = pd.read_csv("dataset/insurance-customer/test.csv")

train_df.drop(["id","Count_3-6_months_late","Count_6-12_months_late","Count_more_than_12_months_late"], axis=1, inplace=True)
test_df.drop(["id","Count_3-6_months_late","Count_6-12_months_late","Count_more_than_12_months_late"], axis=1, inplace=True)

train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

print(train_df)
print(train_df["mocked_target"].value_counts())

train_df["residence_area_type"] = LabelEncoder().fit_transform(train_df["residence_area_type"])
train_df["sourcing_channel"] = LabelEncoder().fit_transform(train_df["sourcing_channel"])
test_df["residence_area_type"] = LabelEncoder().fit_transform(test_df["residence_area_type"])
test_df["sourcing_channel"] = LabelEncoder().fit_transform(test_df["sourcing_channel"])

train_df[].apply()

train_df.to_csv("clean_train.csv",index=False)

train = np.array(train_df.values)
test = np.array(test_df.values)

# np.random.shuffle(train)

x_train = train[:, :-2].astype(float)
y_train = train[:, -1].astype(int)
x_test = test[:, :-2].astype(float)
y_test = test[:, -1].astype(int)

# have to clean data, make groups, bands for incomes, etc
print(x_train)
# x_train = StandardScaler().fit_transform(x_train)
# x_test = StandardScaler().fit_transform(x_test)
# print(x_train)
# x_train = Normalizer().fit_transform(x_train)
# x_test = Normalizer().fit_transform(x_test)
# print(x_train)

# print(x_train.shape)
# print(y_train.shape)



# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(100, activation="tanh"),
#     tf.keras.layers.Dense(1, activation="sigmoid"),
# ])



# model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
#               loss=tf.keras.losses.BinaryCrossentropy(),
#               metrics=tf.keras.metrics.Accuracy(name="accuracy")
#               )

# model.build((76879, 8))
# print(model.summary())

# model.fit(x_train, y_train, epochs=10, batch_size=32)
# # model.evaluate(x_test,  y_test, verbose=2)

# # not learning

# # random forest?
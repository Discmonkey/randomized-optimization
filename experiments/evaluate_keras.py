import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np


num_columns = 18

model_path = "/home/biometrics/BiometricsHg/sandbox/mgrinchenko/randomized-optimization/experiments/results/reduced_set_lr_0001_no_momentumsave_model"
data_path = "/home/biometrics/BiometricsHg/sandbox/mgrinchenko/randomized-optimization/experiments/results/test_fire_reduced.csv"

data = pd.read_csv(data_path)

x, y = data.values[:, 1:num_columns + 1], data.values[:, 0:1]
enc = OneHotEncoder()
enc.fit(y.reshape(-1, 1))

new_y = enc.transform(y).toarray()


model = tf.keras.models.load_model(model_path)


evaluations = np.argmax(model.predict(x, batch_size=5), axis=1)

correct_positives = 0
incorrect_positives = 0
correct_negatives = 0
incorrect_negatives = 0

for actual, result in zip(y, evaluations):
    if actual == 1 and result == 1:
        correct_positives += 1

    if actual == 1 and result == 0:
        incorrect_positives += 1

    if actual == 0 and result == 0:
        correct_negatives += 1

    if actual == 0 and result == 1:
        incorrect_negatives += 1

print correct_positives, incorrect_positives, correct_negatives, incorrect_negatives
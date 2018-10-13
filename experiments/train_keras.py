import tensorflow
from tensorflow import keras
import pandas as pd
from top_level_file import base
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import datetime
import numpy as np

df = pd.read_csv(os.path.join(base, "..", "datasets/cache/train_fire_reduced.csv"))
num_columns = 18
model = keras.models.Sequential()
model.add(keras.layers.InputLayer(input_shape=(num_columns,)))
model.add(keras.layers.Dense(18, activation='relu'))
model.add(keras.layers.Dense(18, activation='relu'))
model.add(keras.layers.Dense(8, activation='relu'))
model.add(keras.layers.Dense(2, activation='softmax'))

optimizer = keras.optimizers.SGD(lr=.001, momentum=.1)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model_dir = os.path.join(base, "training", "fire", "reduced_set")

os.makedirs(model_dir)

x, y = df.values[:, 1:num_columns + 1], df.values[:, 0:1]
enc = OneHotEncoder()
enc.fit(y.reshape(-1, 1))

new_y = enc.transform(y).toarray()

model.fit(x, new_y, batch_size=64, epochs=10000, shuffle=True, validation_split=.1, callbacks=[
    keras.callbacks.TensorBoard(log_dir=model_dir, batch_size=64, write_graph=True)])
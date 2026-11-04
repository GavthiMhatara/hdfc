import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import SGD 
from tensorflow.keras.models import Sequential 
from sklearn.model_selection import train_test_split


data = pd.read_csv('cifar-10.csv')

x = data.drop('label', axis = 1).values
y = data['label'].values

x = x / 255.0;

x = x.reshape(-1, 32, 32, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


model = Sequential([
    Flatten(input_shape=(32,32,3)),
    Dense(512, activation="relu"),
    Dense(256, activation="relu"),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")
])

model.summary()

model.compile(
    optimizer = SGD(learning_rate=(0.01)), 
    loss = "sparse_categorical_crossentropy",
    metrics = ['accuracy']
)

history = model.fit(
    x_train, 
    y_train, 
    epochs=10,
    batch_size=32,
    validation_split=0.2 
)


loss, accuracy = model.evaluate(x_test, y_test);

print(f"Loss : {loss}, Accuracy : {accuracy}")










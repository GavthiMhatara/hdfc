
# 2A. CNN Model for CIFAR-10 Image Classification

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam


# -----------------------------------------
# A. Loading and Preprocessing Image Data
# -----------------------------------------

# Load local dataset
data = pd.read_csv("cifar-10.csv")

# Separate features and labels
X = data.drop("label", axis=1).values
y = data["label"].values

# Reshape images into 32x32 RGB images
X = X.reshape(-1, 32, 32, 3)

# Normalize pixel values
X = X.astype("float32") / 255.0

# Split training and testing data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# -----------------------------------------
# B. Defining CNN Model Architecture
# -----------------------------------------

model = Sequential([
    Input(shape=(32, 32, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.summary()


# -----------------------------------------
# C. Training the CNN Model
# -----------------------------------------

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2
)


# -----------------------------------------
# D. Estimating Model Performance
# -----------------------------------------

loss, accuracy = model.evaluate(X_test, y_test)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy * 100, "%")

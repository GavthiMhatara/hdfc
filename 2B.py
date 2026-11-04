# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import SGD


# -----------------------------------------
# A. Loading and Preprocessing Image Data
# -----------------------------------------

# Load local CSV files
train_data = pd.read_csv("mnist_train.csv")
test_data = pd.read_csv("mnist_test.csv")

# Separate labels and pixel values
X_train = train_data.drop("label", axis=1).values
y_train = train_data["label"].values

X_test = test_data.drop("label", axis=1).values
y_test = test_data["label"].values

# Reshape images into 28x28 grayscale images
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Normalize pixel values between 0 and 1
X_train = X_train  / 255.0
X_test = X_test / 255.0

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# -----------------------------------------
# B. Defining CNN Model Architecture
# -----------------------------------------

model = Sequential([
    Input(shape=(28, 28, 1)),

    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),

    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Display model architecture
model.summary()


# -----------------------------------------
# C. Training the CNN Model
# -----------------------------------------

model.compile(
    optimizer=SGD(learning_rate=0.01),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)


# -----------------------------------------
# D. Estimating Model Performance
# -----------------------------------------

# Evaluate model on testing data
loss, accuracy = model.evaluate(X_test, y_test)

print("\nTest Loss:", loss)
print("Test Accuracy:", accuracy * 100, "%")


# -----------------------------------------
# Display Sample Predictions
# -----------------------------------------

predictions = model.predict(X_test[:5])

for i in range(5):

    predicted_label = np.argmax(predictions[i])

    print("Actual Label:", y_test[i])
    print("Predicted Label:", predicted_label)

    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title("Predicted: " + str(predicted_label))
    plt.axis('off')
    plt.show()


# -----------------------------------------
# Plot Training and Validation Accuracy
# -----------------------------------------

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title('CNN Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
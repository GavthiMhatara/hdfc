
# 1. Import necessary packages

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# 2. Load training and testing data from local CSV files

train_data = pd.read_csv("sign_mnist_train.csv")
test_data = pd.read_csv("sign_mnist_test.csv")


# Separate features and labels

x_train = train_data.drop('label', axis=1).values
y_train = train_data['label'].values

x_test = test_data.drop('label', axis=1).values
y_test = test_data['label'].values


print("Training Data Shape:", x_train.shape)
print("Testing Data Shape:", x_test.shape)


# Normalize the pixel values

x_train = x_train / 255.0
x_test = x_test / 255.0


# Reshape images into 28x28

x_train = x_train.reshape(-1, 28, 28)
x_test = x_test.reshape(-1, 28, 28)


# 3. Define the FNN architecture

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(25, activation='softmax')
])

model.summary()


# 4. Compile and train the model using SGD

model.compile(
    optimizer=SGD(learning_rate=0.01),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train,
    y_train,
    epochs=11,
    batch_size=32,
    validation_split=0.2
)


# 5. Evaluate the network

loss, accuracy = model.evaluate(x_test, y_test)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)


# 6. Plot training loss and accuracy

plt.figure(figsize=(12, 5))


# Accuracy plot

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()


# Loss plot

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()


# 7. Predict sample images

predictions = model.predict(x_test[:5])

for i in range(5):

    predicted_class = np.argmax(predictions[i])

    plt.imshow(x_test[i], cmap='gray')

    plt.title(
        f"Predicted: {predicted_class}, Actual: {y_test[i]}"
    )

    plt.axis('off')
    plt.show()

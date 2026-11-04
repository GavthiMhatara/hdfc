import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Loading and Preprocessing Data
IMG_SIZE = 224
BATCH_SIZE = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    "caltech-101",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    "caltech-101",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# Choose the Pre-trained Model (Transfer Learning)
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

# Freeze Base Layers and Add Custom Layers
base_model.trainable = False


# Add Classification Layers
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(
        train_generator.num_classes,
        activation='softmax'
    )
])

# Model Summary
model.summary()

# Object Detection / Classification Training
# Compile Model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train Model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# f) Compute Loss
train_loss = history.history['loss']
val_loss = history.history['val_loss']

print("Training Loss:", train_loss[-1])
print("Validation Loss:", val_loss[-1])

# g) Evaluate the Model
loss, accuracy = model.evaluate(
    val_generator
)

print("Loss :", loss)
print("Accuracy :", accuracy)

# h) Visualize the Prediction
# h) Visualize the Prediction

images, labels = next(val_generator)
predictions = model.predict(images)
class_names = list(val_generator.class_indices.keys())

plt.figure(figsize=(10, 10))

for i in range(9):
    plt.subplot(3, 3, i+1)

    plt.imshow(images[i])

    actual = class_names[np.argmax(labels[i])]
    predicted = class_names[np.argmax(predictions[i])]

    plt.title(f"Actual: {actual}\nPredicted: {predicted}")
    plt.axis("off")

plt.tight_layout()
plt.show()

# i) Fine Tune the Model
# Unfreeze top layers:
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Recompile:
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Retrain 
fine_tune_history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5
)


# Plot Accuracy and Loss
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)

plt.plot(history.history['accuracy'],label='Training Accuracy')
plt.plot(history.history['val_accuracy'],label='Validation Accuracy')
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'],label='Training Loss')
plt.plot(history.history['val_loss'],label='Validation Loss')

plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()
# =====================================================
# 5C. Transfer Learning using VGG16 on Fashion MNIST
# =====================================================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# =====================================================
# a. Loading and Preprocessing Data
# =====================================================

DATASET_PATH = "fashion mnist"

IMG_SIZE = 224
BATCH_SIZE = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# =====================================================
# b. Choose Pre-trained Model
# =====================================================

base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

# =====================================================
# c. Freeze Base Layers
# =====================================================

for layer in base_model.layers:
    layer.trainable = False

# =====================================================
# d. Add Custom Layers
# =====================================================

model = Sequential([    
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])

model.summary()

# =====================================================
# e. Training
# =====================================================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# =====================================================
# f. Compute Loss
# =====================================================

train_loss = history.history['loss'][-1]
val_loss = history.history['val_loss'][-1]

print("Training Loss :", train_loss)
print("Validation Loss :", val_loss)

# =====================================================
# g. Evaluate Model
# =====================================================

loss, accuracy = model.evaluate(val_data)

print("Test Loss :", loss)
print("Test Accuracy :", accuracy)

# =====================================================
# h. Visualize Predictions
# =====================================================

class_names = list(train_data.class_indices.keys())

images, labels = next(val_data)

predictions = model.predict(images)

plt.figure(figsize=(10,10))

for i in range(9):

    plt.subplot(3,3,i+1)

    plt.imshow(images[i])

    predicted_class = np.argmax(predictions[i])

    plt.title(
        class_names[predicted_class]
    )

    plt.axis('off')

plt.tight_layout()
plt.show()

# =====================================================
# Accuracy Graph
# =====================================================

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# =====================================================
# Loss Graph
# =====================================================

plt.subplot(1,2,2)

plt.plot(
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# =====================================================
# i. Fine Tuning
# =====================================================

for layer in base_model.layers[-4:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

fine_tune_history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

# =====================================================
# Final Evaluation
# =====================================================

loss, accuracy = model.evaluate(val_data)

print("Fine Tuned Loss :", loss)
print("Fine Tuned Accuracy :", accuracy)
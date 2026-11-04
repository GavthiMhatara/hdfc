import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load Dataset
data = pd.read_csv("IMDB Dataset.csv")

# Convert review to words
data["review"] = data["review"].str.lower()
data["tokens"] = data["review"].apply(lambda x: x.split())

# Convert sentiment to numbers
data["sentiment"] = data["sentiment"].map({
    "positive": 1,
    "negative": 0
})

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    data["tokens"],
    data["sentiment"],
    test_size=0.2,
    random_state=42
)

# CBOW Model (sg=0)
cbow = Word2Vec(
    sentences=X_train,
    vector_size=100,
    window=5,
    min_count=1,
    sg=0
)

# Document Embedding
def doc_vector(doc):
    words = [cbow.wv[word] for word in doc if word in cbow.wv]

    if len(words) == 0:
        return np.zeros(100)

    return np.mean(words, axis=0)

# Convert reviews to vectors
X_train_vec = np.array([doc_vector(doc) for doc in X_train])
X_test_vec = np.array([doc_vector(doc) for doc in X_test])

# Feed Forward Neural Network
model = Sequential([
    Dense(128, activation='relu', input_shape=(100,)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    X_train_vec,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2
)

# Evaluate
loss, accuracy = model.evaluate(
    X_test_vec,
    y_test
)

print("Loss :", loss)
print("Accuracy :", accuracy)
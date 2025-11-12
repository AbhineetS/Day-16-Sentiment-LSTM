"""
run_sentiment_lstm.py
LSTM-based Sentiment Classifier using IMDB dataset
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report

# Config
NUM_WORDS = 10000
MAX_LEN = 200
EMBED_DIM = 128
BATCH_SIZE = 128
EPOCHS = 4
MODEL_FILE = "sentiment_lstm.h5"

def load_and_preprocess():
    print("📦 Loading IMDB dataset...")
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=NUM_WORDS)
    print(f"Train samples: {len(x_train)} | Test samples: {len(x_test)}")
    x_train = pad_sequences(x_train, maxlen=MAX_LEN, padding='post', truncating='post')
    x_test = pad_sequences(x_test, maxlen=MAX_LEN, padding='post', truncating='post')
    return x_train, y_train, x_test, y_test

def build_model():
    model = Sequential([
        Embedding(NUM_WORDS, EMBED_DIM, input_length=MAX_LEN),
        LSTM(128, return_sequences=False),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def plot_history(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.plot(history.history['loss'], label='Train Loss', linestyle='--')
    plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--')
    plt.title('Training History (Accuracy & Loss)')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    plt.close()
    print("📈 Saved training history as training_history.png")

def evaluate_model(model, x_test, y_test):
    print("🔍 Evaluating model on test set...")
    y_pred_prob = model.predict(x_test, verbose=0)
    y_pred = (y_pred_prob >= 0.5).astype(int)

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()
    print("✅ Saved confusion_matrix.png")

def main():
    x_train, y_train, x_test, y_test = load_and_preprocess()
    model = build_model()
    es = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

    print("🚀 Training model...")
    history = model.fit(
        x_train, y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[es],
        verbose=2
    )

    plot_history(history)
    print("💾 Saving model...")
    model.save(MODEL_FILE)
    print(f"✅ Model saved as {MODEL_FILE}")

    evaluate_model(model, x_test, y_test)
    print("🎯 Done!")

if __name__ == "__main__":
    main()
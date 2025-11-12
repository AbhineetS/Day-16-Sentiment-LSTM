🧠 **Day 16 — Sentiment Analysis using Deep Learning (LSTM)**

This project focuses on classifying movie reviews as **positive** or **negative** using a **Long Short-Term Memory (LSTM)** network trained on the **IMDB dataset**.  
It demonstrates how recurrent neural networks can capture **sequential word dependencies** to understand sentiment and context in natural language.

---

## 🚀 Overview

- Built an **LSTM-based text classifier** using TensorFlow + Keras  
- Trained on **50,000 IMDB reviews** (25k train + 25k test)  
- Used **word embeddings** to represent reviews numerically  
- Achieved **~77% test accuracy** with a simple single-layer LSTM  
- Visualized training and validation performance with Matplotlib  

---

## 🧠 Workflow

1. **Data Loading & Preprocessing** — Loaded IMDB reviews and tokenized them into integer sequences  
2. **Sequence Padding** — Padded/truncated sequences to uniform length for LSTM input  
3. **Model Design** — Constructed an Embedding → LSTM → Dropout → Dense(Sigmoid) pipeline  
4. **Training & Validation** — Used binary crossentropy and Adam optimizer with validation split  
5. **Evaluation & Visualization** — Generated a confusion matrix and classification report  

---

## 📊 Results

| Metric | Value |
|--------|--------|
| **Accuracy** | 77.1% |
| **Precision (Positive)** | 0.82 |
| **Recall (Positive)** | 0.70 |
| **F1-score (Positive)** | 0.75 |

📈 Training progress and evaluation visuals were saved as:
- `training_history.png`  
- `confusion_matrix.png`

💾 Final model: `sentiment_lstm.h5`

---

## 🧩 Tech Stack

Python | TensorFlow | Keras | NumPy | Matplotlib | Seaborn | Scikit-learn  

---

## 🧠 Key Concepts

- **Recurrent Neural Network (RNN):** A neural network architecture designed to handle sequential data.  
- **LSTM:** A special kind of RNN capable of learning long-term dependencies using *gates* (input, forget, output).  
- **Word Embedding:** A dense vector representation of words, capturing semantic relationships.  
- **Sequence Padding:** Ensures all input sequences have equal length for batch processing.  

---

## 🔗 Connect

💼 [LinkedIn - Abhineet Singh](https://www.linkedin.com/in/abhineet-s)  
📁 [GitHub Repository](https://github.com/AbhineetS/Day-16-Sentiment-LSTM)
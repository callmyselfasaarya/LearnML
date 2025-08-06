# 📩 Spam/Fraudulent Message Detection using Supervised Learning

## 📝 Description

This project aims to detect spam or fraudulent messages using supervised machine learning algorithms such as **Logistic Regression** and **Random Forest**. It utilizes a labeled dataset of messages and converts textual data into numerical features using **TF-IDF vectorization**. The goal is to build and evaluate models that can accurately classify messages as *spam* or *ham* (not spam).

---

## 🔍 Features

- Preprocessing of text data (lowercasing, punctuation removal)
- Feature extraction using **TF-IDF**
- Model training with:
  - Logistic Regression
  - Random Forest Classifier
- Performance evaluation using:
  - Accuracy
  - Precision, Recall, F1 Score
  - Confusion Matrix visualization

---

## 🛠️ Tech Stack

- **Language:** Python 3.x  
- **Libraries:**
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`

---

## 📁 Dataset

- **Source:** [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- Format: Labeled messages as `ham` (non-spam) or `spam`.

---

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/spam-detector.git
   cd spam-detector

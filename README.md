# AI-Powered Smart Email Classifier for Enterprises 📧🤖

**AI Email Classifier** is a machine learning project that automatically categorizes emails into meaningful groups such as **Complaint, Feedback, Support, Spam, and Other**.

The system uses **Natural Language Processing (NLP)** techniques with **TF-IDF vectorization** and a **Support Vector Machine (SVM)** model to analyze email text and predict its category.

This project demonstrates a complete **end-to-end machine learning workflow**, including data preprocessing, model training, evaluation, and building a simple interactive interface using **Streamlit**.

---

# 🚀 Features

- Automatic email classification into **five categories**:

  - **Complaint**
  - **Feedback**
  - **Support**
  - **Spam**
  - **Other**

- Machine learning based text classification using **SVM**

- **TF-IDF feature extraction** for processing email text

- Displays **prediction confidence scores**

- Provides **basic analytics and insights** about classified emails

- Simple **Streamlit interface** for testing email predictions

---

# 🛠 Tech Stack

| Component | Technology |
|----------|------------|
| **Programming Language** | Python |
| **Machine Learning** | Support Vector Machine (SVM) |
| **Text Processing** | TF-IDF Vectorization |
| **Libraries** | Scikit-learn, Pandas, NumPy |
| **UI** | Streamlit |

---


---

# ⚙️ How It Works

## 1. Data Preprocessing

Email text is **cleaned and normalized** by removing unnecessary characters and formatting the text for analysis.

---

## 2. Feature Extraction

The cleaned text is converted into numerical features using:

**TF-IDF (Term Frequency – Inverse Document Frequency)**

This method helps identify important words in the email text.

---

## 3. Model Training

A **Support Vector Machine (SVM)** classifier is trained on the processed dataset to learn patterns from the email text.

---

## 4. Prediction

When a new email is entered:

1. The text is **preprocessed**
2. Converted into **TF-IDF features**
3. Passed to the **trained model**
4. The model predicts the **email category and confidence score**

---

# 📊 What This Project Demonstrates

- **Text classification using machine learning**
- **Natural Language Processing (NLP) techniques**
- **TF-IDF feature engineering**
- **Multi-class classification with SVM**
- **Building a simple ML application using Streamlit**

---

# 🔮 Future Improvements

Possible improvements for the project:

- Add **Explainable AI techniques** such as **SHAP or LIME**
- Build a **REST API using FastAPI**
- Store classified emails in a **database**
- Support **multi-language email classification**

---

# 👨‍💻 Author

**Nunna Hemanth Kumar**


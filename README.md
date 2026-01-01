# 🎬 Sentiment Analysis of Movie Reviews Using Support Vector Machine (SVM)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-SVM-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview

This project implements a complete **Machine Learning pipeline** to classify IMDB movie reviews as **Positive** or **Negative**.

The core objective is to compare the performance of different **Support Vector Machine (SVM)** kernels — **Linear**, **RBF**, and **Polynomial** — on high-dimensional textual data. Unlike many prior implementations where Polynomial kernels fail due to instability, this project introduces a **carefully designed TF-IDF normalization and preprocessing pipeline**, enabling even traditionally unstable kernels to achieve **high accuracy (>86%)**.

The final trained model is deployed as a lightweight and interactive **Streamlit web application** for real-time sentiment prediction.

---

## 📑 Table of Contents

* [Key Features](#-key-features)
* [Methodology](#-methodology)

  * [Data Source](#data-source)
  * [Data Preprocessing](#data-preprocessing)
  * [Feature Extraction](#feature-extraction)
  * [Model Training](#model-training)
* [Performance Results](#-performance-results)
* [Installation & Usage](#-installation--usage)
* [Project Structure](#-project-structure)
* [References](#-references)
* [License](#-license)

---

## 🚀 Key Features

* **Custom Tokenization**: Preserves exclamation marks (`!`) as independent tokens to capture sentiment intensity.
* **Kernel Comparison**: Empirical comparison of Linear, RBF, and Polynomial SVM kernels.
* **Polynomial Kernel Optimization**: Successfully stabilized Polynomial kernels (degree 2 & 3) using L2-normalized TF-IDF vectors.
* **Interactive UI**: Minimal and user-friendly Streamlit interface for live sentiment inference.

---

## 🔬 Methodology

### Data Source

The dataset used is the **Stanford Large Movie Review Dataset (IMDb)**, consisting of **50,000 labeled reviews**, evenly split into **25,000 training** and **25,000 testing** samples.

---

### Data Preprocessing

To reduce noise while retaining critical sentiment signals, the following preprocessing steps were applied:

1. **HTML Tag Removal** – Cleans web-scraped artifacts such as `<br />` tags.
2. **Lowercasing** – Standardizes all text to reduce vocabulary size.
3. **Exclamation Mark Preservation** – Converts expressions like `"Good!"` → `"Good !"` to treat `!` as a meaningful token.
4. **Stemming** – Uses `SnowballStemmer` to reduce words to their root forms (e.g., *loved* → *love*).

---

### Feature Extraction

* **Technique**: TF-IDF (Term Frequency–Inverse Document Frequency)
* **Vocabulary Size**: Limited to the top **5,000 features** to avoid overfitting and memorization of movie titles.
* **Normalization**: L2 normalization applied to stabilize SVM optimization, especially for non-linear kernels.

---

### Model Training

Three SVM classifiers were trained using **scikit-learn**:

1. **Linear Kernel** – Strong baseline for sparse, high-dimensional text data.
2. **RBF Kernel** – Captures non-linear sentiment patterns (**Best Performing Model**).
3. **Polynomial Kernel** – Used as a stress-test model and stabilized via feature scaling.

---

## 📊 Performance Results

| Model Kernel                    | Accuracy   | Precision | Recall | F1-Score |
| ------------------------------- | ---------- | --------- | ------ | -------- |
| **RBF (Radial Basis Function)** | **87.34%** | 0.87      | 0.88   | 0.87     |
| **Linear**                      | 87.18%     | 0.87      | 0.87   | 0.87     |
| **Polynomial**                  | 86.93%     | 0.87      | 0.86   | 0.86     |

**Key Insights:**

* The **RBF kernel** achieved the highest accuracy, indicating subtle non-linear sentiment patterns in text data.
* The **Polynomial kernel**, often unstable in similar research, performed competitively due to strict feature normalization.

---

## 💻 Installation & Usage

### Prerequisites

* Python 3.8+
* pip

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/sahtiwk/Sentiment-Analysis-on-IMDB-movie-reviews.git
cd Sentiment-Analysis-on-IMDB-movie-reviews
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

The application will open in your browser at **[http://localhost:8501](http://localhost:8501)**.

---

## 📂 Project Structure

```
imdb-sentiment-svm/
├── app.py                          # Streamlit application
├── svm_rbf_sentiment_model.pkl     # Trained RBF SVM model
├── tfidf_vectorizer.pkl            # Fitted TF-IDF vectorizer
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── notebook/
    └── Project.ipynb # Training & experimentation notebook
```

---

## 📚 References

* Maas, A. L., et al. *Learning Word Vectors for Sentiment Analysis*. ACL, 2011.
* Govindarajan, M. *Sentiment Analysis of Movie Reviews Using Hybrid Methods*. IJACR, 2013.
* Farasalsabila, F., et al. *Sentiment Analysis for IMDb Movie Reviews Using SVM*. Inform Journal, 2023.
* Steinke, I., et al. *Sentiment Analysis of Online Movie Reviews Using Machine Learning*. IJACSA, 2022.
* Pouransari, H., Ghili, S. *Deep Learning for Sentiment Analysis of Movie Reviews*. Stanford Reports, 2015.

---

## 📝 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

⭐ If you found this project useful, consider starring the repository!

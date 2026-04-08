# 💳 Credit Card Fraud Detection using Machine Learning

## 📌 Project Overview

This project focuses on detecting fraudulent credit card transactions using **Machine Learning**.
The model is trained to classify transactions as **legitimate (0)** or **fraudulent (1)** using **Logistic Regression**.

Fraud detection is crucial in the financial sector to prevent losses and ensure secure transactions.

---

## 🎯 Objectives

* Detect fraudulent transactions accurately
* Handle highly **imbalanced dataset**
* Build a simple and efficient ML model
* Evaluate model performance using accuracy

---

## 📊 Dataset Information

* Dataset contains credit card transaction details
* Target column: **Class**

  * `0` → Normal Transaction
  * `1` → Fraudulent Transaction
* Dataset is **highly imbalanced** (very few fraud cases)

---

## ⚙️ Technologies Used

* Python 🐍
* NumPy
* Pandas
* Scikit-learn

---

## 🔄 Project Workflow

### 1. Data Loading

* Dataset loaded using Pandas
* Initial inspection using `.head()`, `.tail()`, `.info()`

### 2. Data Preprocessing

* Checked for missing values
* Analyzed class distribution
* Observed dataset imbalance

### 3. Data Balancing (Under-Sampling)

* Separated:

  * Legit transactions
  * Fraud transactions
* Randomly sampled legit transactions to match fraud count
* Created a balanced dataset

### 4. Feature & Target Split

* Features (X): All columns except `Class`
* Target (Y): `Class` column

### 5. Train-Test Split

* 80% Training data
* 20% Testing data
* Used `stratify` to maintain class balance

### 6. Model Training

* Algorithm: **Logistic Regression**
* Model trained using training dataset

### 7. Model Evaluation

* Predictions made on:

  * Training data
  * Testing data
* Evaluated using **Accuracy Score**

---

## 📈 Results

* Achieved good accuracy on both training and test data
* Model successfully classifies fraud and normal transactions

---

## ⚠️ Limitations

* Accuracy alone is not sufficient for imbalanced datasets
* Logistic Regression may not capture complex patterns

---

## 🚀 Future Scope

* Use advanced models like:

  * Random Forest
  * XGBoost
  * Neural Networks
* Apply **SMOTE** instead of under-sampling
* Implement real-time fraud detection system

---

## 📚 Conclusion

This project demonstrates how Machine Learning can be used to detect fraudulent transactions effectively.
With further improvements, it can be deployed in real-world banking systems.

---

## 👤 Author

**Iqra Ahenger**
Ramrao Adik Institute of Technology
Navi Mumbai

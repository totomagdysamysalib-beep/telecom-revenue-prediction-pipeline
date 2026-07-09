# 📞 Telecom Customer Churn & Revenue Prediction

A comprehensive end-to-end Machine Learning pipeline developed to analyze telecom customer data, predict customer churn, and forecast total revenue using advanced classification, regression, and hyperparameter tuning techniques.

---

## 🚀 Project Overview

This project was built entirely within the **Google Colab** environment using **Python**. The pipeline processes telecom customer features through two distinct machine learning tracks:
1. **Classification Task:** Predicting the probability of a customer leaving the service (`Churn`).
2. **Regression Task:** Forecasting the total expenses/revenue of a customer (`Total Charges`) while identifying the core driving features behind the company's revenue.

---

## 🛠️ Technologies & Libraries Used

- **Language:** Python 3
- **Data Manipulation:** Pandas, NumPy
- **Data Visualization:** Matplotlib
- **Machine Learning & Analytics:** - `Scikit-Learn` (Preprocessing, Decomposition, Selection, Metrics)
  - `XGBoost` (XGBClassifier, XGBRegressor)

---

## 📋 Pipeline Architecture & Steps

### 1. Data Preprocessing & Cleaning
* **Data Exploration:** Inspected data quality, data types, and structural integrity using `dataset.info()` and `dataset.head()`.
* **Handling Missing Values:** Cleaned and converted the `TotalCharges` column into a numeric format while coercing errors (`errors='coerce'`), replacing missing elements with zeros via `fillna(0)`.
* **Categorical Encoding:** Looped through multi-class textual features using `LabelEncoder`, and utilized `OneHotEncoder` via `ColumnTransformer` for advanced handling of nominal features to ensure optimal compatibility with the models.
* **Data Splitting:** Divided features and target arrays into 80% training and 20% testing subsets using `train_test_split`.
* **Feature Scaling:** Applied `StandardScaler` to standardize numerical ranges, maximizing model convergence speed and training stability.

### 2. Feature Engineering & Dimensionality Reduction
* Implemented **PCA (Principal Component Analysis)** to reduce data dimensionality down to two primary components (`n_components=2`), optimizing computational efficiency while preserving the most vital structural variances.

### 3. Classification Models & Hyperparameter Tuning
Trained and compared multiple robust classifiers, optimized through automated **GridSearchCV** backed by a 10-Fold Cross-Validation strategy:
* **Logistic Regression:** Utilized as a baseline linear model for initial performance benchmarking.
* **Random Forest Classifier:** Fine-tuned over structural parameters including tree criteria, maximum depth, and estimator counts.
* **XGBoost Classifier:** Deployed to capture complex nonlinear boundaries and boost overall predictive accuracy.
* **Evaluation:** Modeled performance using `Confusion Matrix` and `Accuracy Score` metrics to determine the optimal deployment candidate.

### 4. Regression Modeling & Revenue Forecasting
* Isolated features, dropping descriptive parameters like `customerID`, and assigned `TotalCharges` as a continuous target variable for the **XGBoost Regressor**.
* Optimized parameters through a secondary `GridSearchCV` loop, maximizing performance against the `neg_mean_absolute_error` metric.
* **Feature Importances:** Extracted and ordered structural features to reveal the most influential customer traits determining overall bills and business revenue.

---

## 💻 How to Run and Use

1. Upload the notebook file `Untitled25.ipynb` into your **Google Colab** workspace.
2. Ensure your dataset file named `WA_Fn-UseC_-Telco-Customer-Churn.csv` is uploaded to the active working directory.
3. Execute the cells sequentially to observe model training progress, validation tables, and final evaluation metrics.

---

## 📈 Expected Outputs
* High-accuracy performance summaries for each classification model.
* A detailed comparison DataFrame (`comparison_df`) displaying Actual vs. Predicted values for total revenue.
* A descending rank list highlighting the most critical features impacting the business infrastructure.

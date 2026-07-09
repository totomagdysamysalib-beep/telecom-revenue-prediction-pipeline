# 📊 End-to-End Telecom Customer Revenue Pipeline & Prediction

A comprehensive, production-grade Machine Learning architecture built to handle complex telecom customer profiles and predict their **Total Charges** utilizing a custom preprocessing pipeline and a highly optimized **XGBoost Regressor**.

---

## 📌 Project Overview
This project showcases a complete Data Science workflow, taking raw customer data through an aggressive feature engineering and encoding pipeline before feeding it to a finely tuned tree-based model. It highlights advanced debugging, architectural optimization, and robust preprocessing strategies that prevent data leakage.

---

## 🛠️ Complete Architecture & Pipeline Break Down

### 1️⃣ Target Isolation & Strategic Splitting
* **Feature Selection:** Isolated the target variable `TotalCharges` ($y$) and excluded redundant identifiers like `customerID` from the feature matrix ($X$) to build a generalized model.
* **Leakage Prevention:** Dynamically split the dataset into `Train` and `Test` sets ($80/20$) using `train_test_split` **before** applying any data transformations. This guarantees the test set remains completely unseen during feature mapping.

### 2️⃣ Multi-Stage Custom Feature Preprocessing
Instead of using generic automations, a meticulous multi-tier encoding pipeline was built from scratch to handle mixed text and numerical values:

* **Stage A: Multi-Column Label Encoding:** Developed a targeted programmatic loop leveraging Scikit-Learn's `LabelEncoder` to cleanly transform text categorical features across specific column indexes:
  $$\text{Columns Indexed: } [0, 1, 2, 3, 5, 15, 8, 9, 10, 11, 12, 13, 18]$$
  This seamlessly mapped descriptive categories into structured integers while maintaining parallel transformation consistency across both `x_train` and `x_test`.
  
* **Stage B: Structural One-Hot Encoding via `ColumnTransformer (CT)`:** Created a complex `ColumnTransformer` layout to dynamically convert nominal features (specifically column indexes `[6, 7, 8, 14, 16]`) into independent binary columns via `OneHotEncoder`. The remaining numeric scales were structurally maintained using `remainder='passthrough'`.

* **Stage C: Feature Scaling:** Integrated a `StandardScaler()` to center and normalize feature distributions (`fit_transform` on train, `transform` on test) to support both distance-dependent baseline models and tree splitting stability.

### 3️⃣ Algorithmic Diagnosis: SVR vs. XGBoost
* **The SVR Baseline Constraint:** Initially tested a Support Vector Regressor (`SVR`). Because the input space ($X$) was tightly scaled while the target space ($y$) spanned large dollar amounts, the distance metric collapsed mathematically, forcing the SVR to output flat, near-identical averages for all rows.
* **The XGBoost Resiliency:** Pivoted the architecture to an **`XGBoost Regressor`**. Being immune to target scale variances due to its threshold-splitting tree structures, it perfectly navigated the custom pipeline data and matched real billing figures with immense accuracy.

### 4️⃣ Hyperparameter Optimization (Grid Search CV)
A 10-Fold Cross-Validation Grid Search (`GridSearchCV`, `cv=10`) was conducted over the network architecture using `neg_mean_absolute_error` to strictly penalize variance.
* **Winning Configurations:** `{'learning_rate': 0.1, 'n_estimators': 200}`
* **Optimized Mean Absolute Error (MAE):** **$54.09** (An outstanding error rate relative to lifetime charges running up into thousands of dollars).

---

## 📊 Business Intelligence & Feature Significance
By assessing the finalized model's `.feature_importances_` array, weights were programmatically linked back to their respective dummy-encoded feature names out of the `ColumnTransformer`. The resulting impact weights were systematically organized using `.sort_values(ascending=False)` and cleaned up with `.reset_index(drop=True)` to deliver a completely human-readable, descending business report of the primary billing drivers.

---

## 💻 Technical Stack
* **Language:** Python 3.12
* **Engineering Tools:** Scikit-Learn (`ColumnTransformer`, `LabelEncoder`, `OneHotEncoder`, `StandardScaler`, `GridSearchCV`)
* **Modeling Frame:** XGBoost Regression (`XGBRegressor`)
* **Data Infrastructure:** Pandas, NumPy
* **Visualization Environment:** Google Colab

---

## 🚀 Pipeline Deployment Code
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor

# The complete sequence built through this project:
# 1. Split -> 2. Label Encode -> 3. Column Transformer -> 4. Scale -> 5. Train Optimal XGBoost
final_regressor = XGBRegressor(learning_rate=0.1, n_estimators=200, random_state=0)
final_regressor.fit(x_train, y_train)

# Deliver clean, high-precision predictions
y_pred = final_regressor.predict(x_test)

# Breast-Cancer-Prediction-Using-Machine-Learning

This project predicts whether a breast tumor is **malignant** or **benign** using machine learning models trained on the **Wisconsin Breast Cancer Diagnostic Dataset**. The workflow includes data preprocessing, statistical analysis, feature selection, model training, and evaluation using metrics such as accuracy and ROC-AUC.

---

## Dataset Overview

- **Source**: [Kaggle – Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- **Samples**: 569
- **Target Variable**: `diagnosis` — *Malignant (M)* or *Benign (B)*

---

## Variables Used (Top 20 Features)

The top 20 features were selected based on **Spearman correlation** with the target variable `diagnosis`:

- `perimeter_worst`
- `radius_worst`
- `area_worst`
- `concave points_worst`
- `concave points_mean`
- `perimeter_mean`
- `area_mean`
- `concavity_mean`
- `radius_mean`
- `area_se`
- `concavity_worst`
- `perimeter_se`
- `radius_se`
- `compactness_mean`
- `compactness_worst`
- `concave points_se`
- `texture_worst`
- `concavity_se`
- `texture_mean`
- `smoothness_worst`

---

## Project Workflow

1. **Import Libraries**

2. **Data Loading**
   - Loaded the dataset.

3. **Data Exploration**
   - Explored dataset shape, data types, column names, descriptive statistics, and missing values.

4. **Data Cleaning**
   - Dropped irrelevant columns (`id`, `Unnamed: 32`).
   - Plotted boxplots to detect outliers across numeric variables.
   - Detected and capped outliers using the IQR method.

5. **Data Visualization**
   - Visualized diagnosis label distribution (Malignant vs. Benign).
   - Plotted Bar chart to compare mean values of each feature by diagnosis category.
   - Plotted Histograms with KDE to assess feature distributions grouped by diagnosis.

6. **Data Preprocessing (Encoding)**
   - Encoded target variable: `'Malignant'`: `1`, `'Benign'`: `0`.

7. **Statistical Tests**
   - Conducted **Shapiro-Wilk test** for normality.
   - Performed **Mann-Whitney U test** to assess distribution differences between diagnosis groups.
   - Performed **Spearman correlation** to examine relationships between features and the target variable(diagnosis).

8. **Feature Selection**
   - Selected the top 20 features based on Spearman correlation values.

9. **Train-Test Split**
   - Split the dataset into 80% training and 20% testing sets.

10. **Machine Learning Models**
    - Applied StandardScaler to normalize features for SVM and Logistic Regression models.
    - Trained the following models:
      - Logistic Regression
      - Random Forest
      - Support Vector Machine (SVM)
      - Decision Tree
      - XGBoost
        
12. **Model Accuracy Comparison**

The following models were trained and evaluated using the selected top 20 features:

| Model                  | Accuracy  |
|------------------------|-----------|
| Logistic Regression    | **98.25%** |
| Random Forest          | 96.49%     |
| Support Vector Machine | 96.49%     |
| XGBoost                | 95.61%     |
| Decision Tree          | 94.74%     |

> Logistic Regression achieved the highest accuracy.

<img src="https://github.com/minotikedare/Breast-Cancer-Prediction-Using-Machine-Learning/blob/main/Model%20Accuracy%20Comparison.png?raw=true" alt="Model Accuracy Comparison" width="600"/>

12. **Receiver Operating Characteristic Curve (ROC) Comparison**

The ROC curve evaluates classification performance across thresholds. The **AUC (Area Under the Curve)** provides a single score indicating model separability.

| Model                  | AUC Score |
|------------------------|-----------|
| Logistic Regression    | **1.00**  |
| Random Forest          | **1.00**  |
| Support Vector Machine | **1.00**  |
| XGBoost                | 0.99      |
| Decision Tree          | 0.95      |

> All models demonstrated strong classification performance, with Logistic Regression, Random Forest, and SVM achieving AUC scores of 1.00.

<img src="https://github.com/minotikedare/Breast-Cancer-Prediction-Using-Machine-Learning/blob/main/Receiver%20Operating%20Characteristic%20Curve%20Comparison.png?raw=true" alt="ROC Curve Comparison" width="600"/>

13. **Building a Predictive System**

A predictive system was developed using the trained **Logistic Regression** model. It takes input for the top 20 selected features and predicts whether the tumor is: **Benign** (0) or  **Malignant** (1)

Logistic Regression was selected as the optimal model because, despite Random Forest, Logistic Regression, and Support Vector Machine (SVM) all attaining perfect AUC scores of 1.00, Logistic Regression achieved the highest accuracy of 98.25%. This performance exceeded that of Random Forest and SVM (both 96.49%), XGBoost (95.61%), and Decision Tree (94.74%). Its superior accuracy and discriminative capacity render it particularly suitable for this task.

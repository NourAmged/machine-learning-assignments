# Machine Learning Assignments - Level 2, Semester 2

This repository contains three assignments for the Machine Learning course at FCDS, Level 2, Semester 2. Each assignment demonstrates practical applications of machine learning techniques using real-world datasets.

---

## Assignment 1: Regression and Classification

### 1. Housing Price Prediction
**Notebook:** [`Assignment-1/Assignment.ipynb`](Assignment-1/Assignment.ipynb)
- **Dataset:** [Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)
- **Objective:** Predict house prices using regression models.
- **Key Steps:**
  - Exploratory Data Analysis (EDA), outlier handling, feature encoding
  - Simple Linear Regression, Multiple Linear Regression, Polynomial Regression
  - Model evaluation with R² and MSE

### 2. Heart Disease Prediction
**Notebook:** [`Assignment-1/Assignment-1-2.ipynb`](Assignment-1/Assignment-1-2.ipynb)
- **Dataset:** [Framingham Heart Study](https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression/data)
- **Objective:** Predict the 10-year risk of coronary heart disease (CHD) using logistic regression.
- **Key Steps:**
  - Data cleaning, preprocessing, feature scaling
  - Logistic Regression model
  - Model evaluation: accuracy, confusion matrix

---

## Assignment 2: Evaluation Metrics and KNN Classification

**Notebook:** [`Assignment-2/Evaluation-Metrics.ipynb`](Assignment-2/Evaluation-Metrics.ipynb)
- **Dataset:** [Framingham Heart Study](https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression/data)
- **Objective:** Predict heart disease using K-Nearest Neighbors (KNN) and evaluate with various metrics.
- **Key Steps:**
  - Data preprocessing and feature selection
  - KNN model training and hyperparameter tuning
  - Model evaluation: accuracy, confusion matrix, classification report

---

## Assignment 3: Breast Cancer Classification with SVM and Neural Networks

**Notebook:** [`Assignment-3/assignment-3-1.ipynb`](Assignment-3/assignment-3-1.ipynb)
- **Dataset:** Breast Cancer Wisconsin Dataset (`sklearn.datasets.load_breast_cancer`)
- **Objective:** Classify tumors as malignant or benign using SVM and Neural Networks.
- **Key Steps:**
  - Data loading, normalization, EDA
  - SVM with RBF, linear, and polynomial kernels
  - Neural Network (Keras Sequential)
  - Model evaluation: confusion matrix, F1-score, ROC curve, loss curve
  - Visual comparison of SVM and Neural Network performance

---

## Requirements

- Python 3.10+
- Jupyter Notebook
- Libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `tensorflow`, `plotly`

Install dependencies with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow plotly
```

---

## How to Run

1. Open the desired notebook (`.ipynb`) in Jupyter Notebook or VS Code.
2. Run the cells sequentially to reproduce the results and visualizations.
3. Modify parameters or code as needed for experimentation.

---

## Authors

- نور الدين امجد عبد الفتاح احمد محمد علوه (ID: 23011596)
- علي ابوبكر كمال محمود (ID: 23012267)
- عبد الله نجاح محمد حسن (ID: 23011356)

---
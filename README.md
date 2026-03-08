# Telecommunication Customer Churn Prediction

> **Final Year Project** — Minors in AI/ML Programme, DJ Sanghvi College of Engineering

A complete machine learning pipeline for predicting customer churn in the telecommunications industry. The project explores data analysis, feature engineering, model comparison, hyperparameter tuning, and neural network approaches on a real-world dataset of 7,043 customers.

---

## Problem Statement

Customer churn — when subscribers cancel their service — is one of the most expensive problems in telecom. Acquiring new customers costs 5–25x more than retaining existing ones. This project builds predictive models to identify at-risk customers before they leave, enabling targeted retention strategies.

---

## Dataset

| Property | Value |
|---|---|
| Total Customers | 7,043 |
| Features | 21 (demographics, services, account info) |
| Target | Churn (Yes/No) |
| Class Split | ~73.5% No / ~26.5% Yes |

**Feature categories:**
- **Demographics**: Gender, Senior Citizen, Partner, Dependents
- **Services**: Phone, Internet (DSL/Fiber), Online Security, Streaming TV/Movies, Tech Support
- **Account**: Contract type, Payment method, Tenure, Monthly/Total Charges

---

## Methodology

### 1. Exploratory Data Analysis
- Class imbalance analysis (26.5% churners)
- Boxplots revealing churners have **lower tenure** (~10 months median) and **higher monthly charges** (~$80 median)
- Feature-level countplots identifying high-churn segments:
  - Month-to-month contracts
  - Electronic check payments
  - Fiber optic internet (higher cost, higher churn)
  - No online security or tech support

### 2. Feature Engineering & Preprocessing
- Label encoding for binary features, one-hot encoding for multi-class
- StandardScaler normalization for numerical features
- Correlation analysis and feature selection
- 80/20 train-test split

### 3. Baseline Models
- **K-Nearest Neighbors (KNN)**
- **Logistic Regression** (L1/L2 regularization)
- **Random Forest**
- **Support Vector Machine (SVM)**

### 4. Hyperparameter Tuning
- **GridSearchCV** for KNN (optimal k), Logistic Regression (regularization), SVM (C value)
- **RandomizedSearchCV** for Random Forest (n_estimators, max_depth, max_features, criterion)
- Cross-validation to address train-test split bias

### 5. Neural Network
- Feed-forward neural network with Keras
- Architecture: Dense layers (1024 → 768 → 512 → 256 → 128 → 1) with Dropout
- ModelCheckpoint for best model selection
- Training history visualization (loss and accuracy curves)

---

## Evaluation

Models evaluated using metrics suited for imbalanced classification:

- **Accuracy**
- **ROC AUC Score**
- **Precision / Recall / F1 Score**
- **Confusion Matrix**
- **ROC Curve** visualization

### Key Insights

- **Contract type** is the strongest churn predictor — month-to-month customers churn at significantly higher rates
- **Tenure** is inversely correlated with churn — early intervention is critical
- **Fiber optic** customers churn more despite (or because of) higher monthly charges
- Customers without **online security** or **tech support** add-ons are more likely to churn
- Ensemble and tuned models outperform baseline approaches

---

## Tech Stack

- **Python**: pandas, NumPy, scikit-learn
- **Deep Learning**: Keras / TensorFlow
- **Visualization**: Matplotlib, Seaborn (FiveThirtyEight style)
- **Tuning**: GridSearchCV, RandomizedSearchCV
- **Environment**: Jupyter Notebook

---

## Project Structure

```
├── Churn Analysis in Telecommunication.ipynb   # Full analysis notebook
├── Customer Churn Prediction.docx              # Project report
├── telecommunication data.csv                  # Dataset
└── README.md
```

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/Anuj-G-06/Telecommunication-Churn-Prediction.git
cd Telecommunication-Churn-Prediction

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn keras tensorflow

# Open the notebook
jupyter notebook "Churn Analysis in Telecommunication.ipynb"
```

---

## Author

**Anuj Gupta**
- Final Year Project, Minors in AI/ML — DJ Sanghvi College of Engineering
- [LinkedIn](https://www.linkedin.com/in/anuj-gupta-2k/)
- [GitHub](https://github.com/Anuj-G-06)

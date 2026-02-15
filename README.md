# Binary-Classification-with-a-Bank-Dataset





##  Project Overview

This project implements a complete Machine Learning pipeline and predict whether a client will subscribe to a bank term deposit

Competition: Playground Series S5E8  
https://www.kaggle.com/competitions/playground-series-s5e8

The objective is to build and compare multiple classification models using experiment tracking and evaluation metrics.

The project includes:
- Data preprocessing
- Missing value handling
- Feature engineering
- Model training (6 algorithms)
- Hyperparameter tuning
- Model comparison
- Metadata tracking

---

## Dataset

Source: Kaggle – Playground Series S5E8  

Files used:
- `train.csv`



Download via CLI:

```bash
kaggle competitions download -c playground-series-s5e8
````

---

## Data Preprocessing

* Mean imputation applied to numerical features
* Train-fitted imputer used on test data (no leakage)
* Scaling applied where required (SGD, KNN, Naive Bayes)
* Feature consistency maintained across train and test sets

---

##  Models Implemented

### Logistic Regression (SGD)

Parameters

* loss: log_loss
* penalty: l2
* alpha: 0.0001
* learning_rate: optimal
* max_iter: 20
* random_state: 42

Performance

* Accuracy: 0.884
* AUC: 0.6371
* F1: 0.3934
* MCC: 0.3489

---

### Decision Tree

Parameters

* criterion: entropy
* max_depth: 6
* min_samples_split: 20
* min_samples_leaf: 10
* max_features: sqrt
* class_weight: balanced

Performance

* Accuracy: 0.7852
* AUC: 0.8414
* F1: 0.507
* MCC: 0.4788

---

### K-Nearest Neighbors

Parameters

* n_neighbors: 7
* weights: distance
* algorithm: auto
* p: 2

Performance

* Accuracy: 0.9088
* AUC: 0.7329
* F1: 0.5699
* MCC: 0.5262

---

### Random Forest 

Parameters

* n_estimators: 300
* min_samples_split: 10
* min_samples_leaf: 5
* max_features: sqrt
* class_weight: balanced
* n_jobs: 4
* random_state: 42

Performance

* Accuracy: 0.9197
* AUC: 0.887
* F1: 0.7172
* MCC: 0.682

---

### XGBoost

Parameters

* n_estimators: 200
* learning_rate: 0.03
* max_depth: 6
* min_child_weight: 3
* subsample: 0.8
* colsample_bytree: 0.8
* gamma: 0.1
* reg_alpha: 0.1
* reg_lambda: 1.0
* random_state: 42

Performance

* Accuracy: 0.9291
* AUC: 0.7931
* F1: 0.6762
* MCC: 0.641

---

### Gaussian Naive Bayes

Parameters

* var_smoothing: 1e-09

Performance

* Accuracy: 0.8447
* AUC: 0.7917
* F1: 0.5287
* MCC: 0.4676

---

##  Model Comparison

| Model               | Accuracy | AUC    | F1     | MCC    |
| ------------------- | -------- | ------ | ------ | ------ |
| Logistic Regression | 0.884    | 0.6371 | 0.3934 | 0.3489 |
| Decision Tree       | 0.7852   | 0.8414 | 0.507  | 0.4788 |
| KNN                 | 0.9088   | 0.7329 | 0.5699 | 0.5262 |
| Random Forest       | 0.9197   | 0.887  | 0.7172 | 0.682  |
| XGBoost             | 0.9291   | 0.7931 | 0.6762 | 0.641  |
| Naive Bayes         | 0.8447   | 0.7917 | 0.5287 | 0.4676 |

Best overall balance: Random Forest
Highest accuracy: XGBoost

---

##  Experiment Tracking

Each model stores:

* Hyperparameters
* Final evaluation metrics
* Model file path

All tracked inside:

```
model_metadata.json
```

---




##  How to Run

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

**Install Dependencies:**
```bash
pip install -r requirements.txt
```
---

**Submitted by:**  Surya Teja

**ID:** 2024dc04253

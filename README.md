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
## Dataset Description

The dataset consists of 750,000 records and 9 numerical variables, including the target variable y, making it a large-scale binary classification problem. The features include demographic and campaign-related information such as age, account balance, day of contact, call duration, number of campaign contacts (campaign), previous contacts (previous), and pdays (number of days since last contact).

The target variable y is binary (0 or 1), with a mean value of 0.1207, indicating class imbalance (~12% positive class). Some variables such as balance and duration show high variability and contain extreme values, while pdays includes -1 values indicating no previous contact. Proper preprocessing such as handling missing/placeholder values and scaling was required before model training.

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

| Model               | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
| ------------------- | -------- | ------ | --------- | ------ | ------ | ------ |
| Logistic Regression | 0.884    | 0.6371 | 0.5332    | 0.3117 | 0.3934 | 0.3489 |
| Decision Tree       | 0.7852   | 0.8414 | 0.3506    | 0.9156 | 0.507  | 0.4788 |
| KNN                 | 0.9088   | 0.7329 | 0.6607    | 0.5011 | 0.5699 | 0.5262 |
| Random Forest       | 0.9197   | 0.887  | 0.6234    | 0.844  | 0.7172 | 0.682  |
| XGBoost             | 0.9291   | 0.7931 | 0.7527    | 0.6138 | 0.6762 | 0.641  |
| Naive Bayes         | 0.8447   | 0.7917 | 0.4171    | 0.7219 | 0.5287 | 0.4676 |

| ML Model Name                | Observation about Model Performance                                                                                                                                                                                                                                                                                                                            |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Logistic Regression**      | Logistic Regression achieved good overall accuracy (0.884) but had relatively low recall (0.3117), meaning it missed many positive cases. Its moderate precision (0.5332) indicates it makes reasonable positive predictions but struggles with capturing minority patterns. As a linear model, it may not fully capture complex relationships in the dataset. |
| **Decision Tree**            | The Decision Tree achieved very high recall (0.9156), meaning it successfully identified most positive cases. However, its precision (0.3506) is low, indicating many false positives. This suggests the model is aggressive in predicting the positive class and may be slightly overfitting despite depth and split constraints.                             |
| **kNN**                      | kNN produced balanced performance with decent accuracy (0.9088) and moderate F1-score (0.5699). Precision (0.6607) is relatively strong, but recall (0.5011) shows it misses some positives. Performance depends heavily on feature scaling and distance metrics, and it may struggle with high-dimensional data.                                              |
| **Naive Bayes**              | Naive Bayes achieved good recall (0.7219) but relatively low precision (0.4171). This suggests it predicts many positives but also generates false alarms. Since it assumes feature independence, performance may be limited if features are correlated. It performs reasonably well given its simplicity.                                                     |
| **Random Forest (Ensemble)** | Random Forest delivered the best overall balance, achieving high accuracy (0.9197), strong recall (0.844), and the highest F1-score (0.7172) and MCC (0.682). The ensemble approach reduces overfitting and improves generalization, making it the most stable and reliable model for this dataset.                                                            |
| **XGBoost (Ensemble)**       | XGBoost achieved the highest accuracy (0.9291) and strongest precision (0.7527), meaning it is very confident in positive predictions. However, its recall (0.6138) is lower than Random Forest. It performs very well overall and handles complex feature interactions effectively, but slightly underperforms RF in balanced metrics (F1, MCC).              |

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

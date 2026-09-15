````markdown
# Olist MLOps Project

Machine learning project based on the **Brazilian E-Commerce Public Dataset by Olist**.

The project builds a reproducible machine learning pipeline for predicting whether an order will be delivered late.

---

## Project Objective

The objective of this project is to build an end-to-end machine learning workflow for late-delivery prediction.

The project covers:

- Data ingestion from PostgreSQL
- Data validation
- Aggregation and joining
- Target creation
- Train / validation / test splitting
- Exploratory Data Analysis
- Feature engineering
- Missing-value handling
- Encoding and scaling
- Baseline modeling
- Model tuning
- Final evaluation
- Saving preprocessing and model artifacts

The complete workflow is organized into six sequential Jupyter notebooks.

---

## Project Structure

```text
Olist-MLOps/
│
├── notebooks/
│   ├── 01_read_and_join.ipynb
│   ├── 02_create_labels.ipynb
│   ├── 03_train_val_test_split.ipynb
│   ├── 04_eda.ipynb
│   ├── 05_feature_engineering.ipynb
│   └── 06_train_evaluate.ipynb
│
├── scripts/
│   ├── load_data.py
│   ├── test_connection.py
│   └── test_queries.py
│
├── artifacts/
│
├── data/
│
├── compose.yaml
├── requirements.txt
├── .gitignore
├── .gitattributes
└── README.md
````

The following directories are stored locally and are not tracked by Git:

```text
data/
artifacts/
venv/
```

They are excluded because:

* `data/` contains the local Olist dataset files.
* `artifacts/` contains generated pipeline outputs.
* `venv/` contains the local Python virtual environment.

---

# Machine Learning Problem

The supervised learning task is:

> Predict whether an Olist order will be delivered late.

The target variable is:

```text
0 = Delivered on time or early
1 = Delivered late
```

The label is created by comparing:

```text
order_delivered_customer_date
```

with:

```text
order_estimated_delivery_date
```

An order is classified as late when the actual delivery date is later than the estimated delivery date.

---

## Prediction Point

The project assumes that the prediction is made:

> After order approval and before carrier handoff.

This prediction point is important because only information available at prediction time should be used by the model.

Columns containing future information are excluded from the model to prevent data leakage.

Examples include:

```text
order_delivered_carrier_date
order_delivered_customer_date
review-related information
delay_days
actual_delivery_days
```

---

# Dataset

The project uses the Brazilian E-Commerce Public Dataset by Olist.

The source dataset consists of multiple relational tables, including:

```text
orders
customers
order_items
order_payments
order_reviews
products
sellers
geolocation
product_category_translation
```

Important one-to-many tables such as:

```text
order_items
order_payments
order_reviews
```

are aggregated before joining.

The final machine learning table therefore contains:

> One row per order.

---

# Pipeline

The workflow is divided into six notebooks.

Each notebook performs one main task and saves artifacts for later stages.

---

# Notebook 01 — Read and Join

File:

```text
notebooks/01_read_and_join.ipynb
```

## Purpose

This notebook:

* Connects to PostgreSQL
* Reads the Olist tables
* Checks row counts
* Checks primary identifiers
* Checks duplicates
* Examines one-to-many relationships
* Aggregates order items
* Aggregates payments
* Aggregates reviews
* Aggregates seller information
* Adds geographic information
* Creates one final row per order
* Saves the joined dataset

## Final result

```text
Orders in ML table : 99,441
Columns            : 44
Unique order_id    : 99,441
Duplicate order_id : 0
```

## Artifact

```text
artifacts/01_joined/ml_table.parquet
```

---

# Notebook 02 — Create Labels

File:

```text
notebooks/02_create_labels.ipynb
```

## Purpose

This notebook creates the late-delivery target.

Orders without enough information to calculate the label are removed.

## Final result

```text
Input orders       : 99,441
Labelable orders   : 96,470
Removed orders     : 2,971

On-time orders     : 89,936
Late orders        : 6,534

Late percentage    : 6.77%
Imbalance ratio    : 13.76 : 1

Columns            : 46
Duplicate order_id : 0
Missing labels     : 0
```

The target is clearly imbalanced, which affects the metric selection used later in the modeling stage.

## Artifact

```text
artifacts/02_labeled/labeled_table.parquet
```

---

# Notebook 03 — Train / Validation / Test Split

File:

```text
notebooks/03_train_val_test_split.ipynb
```

## Purpose

The dataset is split chronologically rather than randomly.

The split ratio is:

```text
70% Train
15% Validation
15% Test
```

A chronological split better represents a production scenario in which a model is trained using historical data and evaluated on later orders.

## Final result

```text
Input labeled orders : 96,470

Train                : 67,529 (70.00%)
Validation           : 14,470 (15.00%)
Test                 : 14,471 (15.00%)
```

Late-delivery rates:

```text
Train late rate      : 7.83%
Validation late rate : 4.31%
Test late rate       : 4.28%
```

Validation checks:

```text
Order overlap       : 0
Chronological order : Valid
```

## Artifacts

```text
artifacts/03_splits/train.parquet
artifacts/03_splits/validation.parquet
artifacts/03_splits/test.parquet
```

---

# Notebook 04 — Exploratory Data Analysis

File:

```text
notebooks/04_eda.ipynb
```

## Important Rule

EDA is performed on:

```text
TRAIN ONLY
```

Validation and test data are not used during exploratory analysis.

## Analysis Includes

* Dataset structure
* Data types
* Memory usage
* Missing values
* Missing-value relationships with the target
* Constant columns
* Numerical distributions
* Numerical ranges
* Skewness
* Outliers
* Numerical relationships with late delivery
* Categorical cardinality
* Rare categories
* Categorical relationships with the target
* Customer state analysis
* Seller state analysis
* Product category analysis
* Payment type analysis
* Monthly patterns
* Weekday patterns
* Hour-of-day patterns
* Holiday indicators
* Customer-seller distance
* Same-state versus different-state delivery
* ZIP-code patterns
* Correlation analysis
* Leakage analysis

## Training data

```text
Training rows      : 67,529
Late-delivery rate : 7.83%
```

## Example findings

Late orders have a larger average customer-seller distance.

Approximate mean distance:

```text
On-time orders : 599 km
Late orders    : 793 km
```

Late rate when customer and seller are in the same state:

```text
≈ 4.45%
```

Late rate when customer and seller are in different states:

```text
≈ 9.59%
```

The highest reliable monthly late-delivery rate was observed around:

```text
2018-03
Late rate ≈ 18.96%
```

## Outputs

EDA outputs are stored in:

```text
artifacts/04_eda/
```

Important files include:

```text
findings_summary.md
column_profile.csv
missing_summary.csv
missing_impact.csv
numerical_summary.csv
outlier_summary.csv
categorical_summary.csv
monthly_summary.csv
monthly_summary_reliable.csv
distance_target_summary.csv
same_state_summary.csv
correlation_matrix.csv
```

Charts are stored in:

```text
artifacts/04_eda/charts/
```

---

# Notebook 05 — Feature Engineering

File:

```text
notebooks/05_feature_engineering.ipynb
```

## Purpose

This notebook creates the final features used by the machine learning model.

Feature choices are based on the findings from Notebook 04.

Notebook 05 also reads:

```text
artifacts/04_eda/findings_summary.md
```

to maintain the pipeline connection between EDA and feature engineering.

---

## Engineered Features

Examples include:

```text
purchase_month
purchase_weekday
purchase_hour
estimated_delivery_days
approval_hours
is_fixed_national_holiday
same_customer_seller_state
customer_seller_distance_km
```

---

## Numerical Features

The project uses 25 numerical features.

Examples include:

```text
item_count
unique_products
unique_sellers
total_item_price
avg_item_price
total_freight_value
avg_freight_value
unique_product_categories
avg_product_weight_g
max_product_weight_g
payment_records
payment_types_count
payment_total
payment_installments_max
estimated_delivery_days
approval_hours
customer_seller_distance_km
```

---

## Categorical Features

The project uses 6 categorical features:

```text
customer_state
primary_product_category
primary_seller_state
primary_payment_type
purchase_month
purchase_weekday
```

---

## Raw Feature Count

```text
Numeric features     : 25
Categorical features : 6
Raw selected features: 31
```

---

## Preprocessing

### Numerical pipeline

Numerical features use:

```text
Median imputation
Missing-value indicators
StandardScaler
```

### Categorical pipeline

Categorical features use:

```text
Missing-category handling
Rare-category grouping
One-hot encoding
```

The encoder is configured to handle infrequent and previously unseen categories.

---

## Training Rule

The preprocessing pipeline is fitted using:

```text
TRAIN ONLY
```

Validation and test data use:

```text
TRANSFORM ONLY
```

This prevents data leakage.

---

## Final Feature Matrices

```text
Train      : (67,529, 159)
Validation : (14,470, 159)
Test       : (14,471, 159)
```

Validation checks:

```text
NaN values : 0
Inf values : 0
```

The transformed matrices are stored as sparse matrices.

---

## Artifacts

```text
artifacts/05_features/
```

Important files include:

```text
engineered_train.parquet
engineered_validation.parquet
engineered_test.parquet

X_train.npz
X_validation.npz
X_test.npz

y_train.npy
y_validation.npy
y_test.npy

train_order_ids.csv
validation_order_ids.csv
test_order_ids.csv

preprocessor.joblib
feature_names.json
feature_config.json
```

---

# Notebook 06 — Train, Tune and Evaluate

File:

```text
notebooks/06_train_evaluate.ipynb
```

## Purpose

This notebook:

* Loads training and validation feature matrices
* Builds a simple baseline
* Trains an initial model
* Tunes model hyperparameters
* Selects a classification threshold
* Freezes model decisions
* Opens the test set only at the final evaluation stage
* Calculates final metrics
* Saves the trained model and results

---

# Baseline

The baseline model is:

```text
DummyClassifier
```

using the prior class distribution.

Validation Average Precision:

```text
0.043124
```

---

# Machine Learning Model

The selected model is:

```text
Logistic Regression
```

Hyperparameters are selected using the validation split only.

The tested values include different:

```text
C
class_weight
```

values.

---

## Selected Configuration

```text
Model                   : Logistic Regression
Best C                  : 0.01
Best class weight       : None
Classification threshold: 0.082211
```

---

# Evaluation Metric

Because late delivery is an imbalanced classification problem, model selection does not rely only on accuracy.

The primary metric is:

```text
Average Precision
```

which summarizes performance on the Precision-Recall curve.

Additional metrics include:

```text
ROC-AUC
F1-score
Precision
Recall
Balanced Accuracy
Accuracy
```

---

# Validation Results

```text
Baseline Average Precision : 0.043124
Final Average Precision    : 0.116389
ROC-AUC                    : 0.753914
F1-score                   : 0.209774
Precision                  : 0.139461
Recall                     : 0.423077
```

The trained model performs better than the baseline on the validation split.

---

# Final Test Results

The test set is opened only after:

* Feature engineering decisions are fixed
* Model configuration is fixed
* Classification threshold is fixed

Final test results:

```text
Average Precision : 0.076008
ROC-AUC           : 0.636449
F1-score          : 0.106585
Precision         : 0.064440
Recall            : 0.308065
```

---

## Test Confusion Matrix

```text
True Negatives  : 11,078
False Positives : 2,773
False Negatives : 429
True Positives  : 191
```

---

## Generalization Observation

Model performance decreased between the validation period and the later chronological test period.

Average Precision:

```text
Validation : 0.116389
Test       : 0.076008
```

ROC-AUC:

```text
Validation : 0.753914
Test       : 0.636449
```

This suggests weaker generalization to later orders or temporal distribution shift in the dataset.

No additional model tuning was performed using the test results.

---

# Model Artifacts

Notebook 06 saves its outputs inside:

```text
artifacts/06_model/
```

Important files include:

```text
model_bundle.joblib
results_summary.json
results_summary.md
test_metrics.csv
test_confusion_matrix.csv
test_predictions.csv
validation_model_comparison.csv
validation_tuning_results.csv
```

Charts are stored in:

```text
artifacts/06_model/charts/
```

---

# Environment Setup

## 1. Create a Virtual Environment

From the project root:

```powershell
python -m venv venv
```

---

## 2. Activate the Environment

On Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Alternatively, commands can be executed directly using:

```powershell
.\venv\Scripts\python.exe
```

---

## 3. Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

or:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## 4. Check Dependencies

```powershell
.\venv\Scripts\python.exe -m pip check
```

Expected result:

```text
No broken requirements found.
```

---

# Python Dependencies

The project uses:

```text
pandas==3.0.5
numpy==2.5.2
matplotlib==3.10.8
pyarrow==25.0.1
SQLAlchemy==2.0.51
psycopg2-binary==2.9.12
joblib==1.5.3
scipy==1.18.1
scikit-learn==1.9.0
ipykernel==7.3.0
```

---

# PostgreSQL with Docker

PostgreSQL runs inside a Docker container.

Start the database with:

```powershell
docker compose up -d
```

Check the container:

```powershell
docker ps
```

The PostgreSQL container is exposed locally on:

```text
127.0.0.1:5433
```

The Docker container internally uses PostgreSQL port:

```text
5432
```

The host-to-container mapping is therefore:

```text
5433 → 5432
```

---

# Loading the Olist Dataset

Place the Olist CSV files inside:

```text
data/
```

The `data/` directory is intentionally excluded from Git.

Run:

```powershell
.\venv\Scripts\python.exe .\scripts\load_data.py
```

This loads the dataset into PostgreSQL.

---

# Testing the Database Connection

Run:

```powershell
.\venv\Scripts\python.exe .\scripts\test_connection.py
```

Additional test queries can be executed with:

```powershell
.\venv\Scripts\python.exe .\scripts\test_queries.py
```

---

# Running the Machine Learning Pipeline

The notebooks must be executed in this exact order:

```text
01_read_and_join.ipynb
        ↓
02_create_labels.ipynb
        ↓
03_train_val_test_split.ipynb
        ↓
04_eda.ipynb
        ↓
05_feature_engineering.ipynb
        ↓
06_train_evaluate.ipynb
```

For a clean execution:

1. Open the notebook in VS Code.
2. Select the project's `venv` Python kernel.
3. Restart the kernel.
4. Run all cells.
5. Wait for the notebook to finish successfully.
6. Continue to the next notebook.

Each notebook creates artifacts that are used by the following stages.

---

# Artifact Flow

```text
PostgreSQL
    │
    ▼
01_read_and_join
    │
    └── ml_table.parquet
            │
            ▼
02_create_labels
    │
    └── labeled_table.parquet
            │
            ▼
03_train_val_test_split
    │
    ├── train.parquet
    ├── validation.parquet
    └── test.parquet
            │
            ▼
04_eda
    │
    ├── findings_summary.md
    ├── CSV summaries
    └── charts
            │
            ▼
05_feature_engineering
    │
    ├── preprocessor.joblib
    ├── feature_names.json
    ├── feature_config.json
    ├── X_train.npz
    ├── X_validation.npz
    ├── X_test.npz
    ├── y_train.npy
    ├── y_validation.npy
    └── y_test.npy
            │
            ▼
06_train_evaluate
    │
    ├── model_bundle.joblib
    ├── results_summary.md
    ├── results_summary.json
    ├── test_metrics.csv
    ├── test_predictions.csv
    └── validation tuning results
```

---

# Reproducibility

The complete pipeline was tested from a clean `artifacts/` directory.

The following sequence was successfully executed from scratch:

```text
01_read_and_join          ✅
02_create_labels          ✅
03_train_val_test_split   ✅
04_eda                    ✅
05_feature_engineering    ✅
06_train_evaluate         ✅
```

All required artifacts were regenerated successfully.

---

# Portable Project Paths

The notebooks do not use a fixed absolute path such as:

```text
F:\...
```

or:

```text
G:\...
```

The project root is detected dynamically.

This allows the repository to be moved to a different drive or directory without modifying the notebook paths.

The portability was verified by moving the project between drives and rerunning the pipeline successfully.

---

# Data Leakage Prevention

Several precautions are used throughout the project.

## EDA

EDA uses:

```text
TRAIN ONLY
```

## Preprocessing

The preprocessing pipeline is fitted using:

```text
TRAIN ONLY
```

Validation and test data use:

```text
TRANSFORM ONLY
```

## Model Selection

Hyperparameters are selected using:

```text
VALIDATION ONLY
```

## Threshold Selection

The classification threshold is selected using:

```text
VALIDATION ONLY
```

## Final Test

The test set is used only after all model decisions have been finalized.

It is used only for:

```text
FINAL EVALUATION
```

---

# Git Tracking

The repository tracks:

```text
notebooks/
scripts/
compose.yaml
requirements.txt
README.md
.gitignore
.gitattributes
```

The repository does not track:

```text
venv/
data/
artifacts/
```

because these contain local environments, datasets, or reproducible generated outputs.

---

# Final Summary

This project implements a reproducible end-to-end machine learning workflow for late-delivery prediction using the Olist e-commerce dataset.

The pipeline includes:

```text
Database
    ↓
Data Joining
    ↓
Target Creation
    ↓
Chronological Split
    ↓
EDA
    ↓
Feature Engineering
    ↓
Preprocessing
    ↓
Baseline Model
    ↓
Model Tuning
    ↓
Threshold Selection
    ↓
Final Test Evaluation
    ↓
Saved Model and Results
```

The final model is a Logistic Regression classifier selected using validation Average Precision and evaluated once on a chronological test set.

```
```

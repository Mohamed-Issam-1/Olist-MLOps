# Olist MLOps Project

Machine learning project based on the **Brazilian E-Commerce Public Dataset by Olist**.

The project predicts whether an order will be delivered late and then turns the trained model into a production-style inference service.

Task 2 contains the training workflow in six sequential Jupyter notebooks. Task 3 extends the project with reusable Python modules, DVC, Great Expectations, MLflow Model Registry, FastAPI, Docker Compose, automated testing, CI/CD, structured logging, and runtime monitoring.

---

## Project Objective

The project has two main stages:

### Task 2 — Model Development

Build a reproducible machine learning workflow that covers:

- Data ingestion from PostgreSQL
- Data validation
- Aggregation and joining
- Target creation
- Chronological train / validation / test splitting
- Exploratory Data Analysis
- Feature engineering
- Missing-value handling
- Encoding and scaling
- Baseline modeling
- Model tuning
- Threshold selection
- Final evaluation
- Saving fitted preprocessing and model artifacts

### Task 3 — Production Inference

Turn the trained Task 2 model into an inference system that:

- loads the already fitted preprocessing objects
- loads the registered model
- never retrains or refits during inference
- accepts new orders from a CLI and FastAPI
- validates incoming data
- returns prediction, probability, and model version
- logs prediction activity and service behavior
- runs inside Docker containers
- starts from a clean machine with one command
- exposes monitoring metrics and simple drift alerts
- is tested automatically in CI/CD

---

# Task 3 — Production Inference Service

## Clean-Machine Quick Start

### Prerequisites

Install:

```text
Git
Docker Desktop
PowerShell
```

The project uses DVC remote storage on DagsHub.

A DagsHub token is required when the data and model artifacts need to be restored from the remote.

**Never commit the token to Git.**

### Clone the repository

```powershell
git clone https://github.com/Mohamed-Issam-1/Olist-MLOps.git
cd Olist-MLOps
git checkout task3-production
```

### Start the complete system

Run:

```powershell
.\scripts\start.ps1
```

The startup script:

1. checks Docker
2. prepares the local environment configuration
3. securely requests the DagsHub credential when required
4. validates Docker Compose
5. restores DVC-managed data and artifacts
6. starts PostgreSQL
7. starts the MLflow PostgreSQL backend
8. starts the MLflow server
9. bootstraps the registered production model
10. starts the FastAPI inference service
11. waits until the API is healthy

No model training is performed during startup.

When startup succeeds:

```text
API      : http://localhost:8000
API Docs : http://localhost:8000/docs
MLflow   : http://localhost:5000
```

Check the service:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

---

## Current Repository Structure

```text
Olist-MLOps/
|
|-- app/
|   |-- __init__.py
|   |-- main.py
|   `-- schemas.py
|
|-- config/
|   `-- config.json
|
|-- src/
|   `-- olist_ml/
|       |-- __init__.py
|       |-- artifacts.py
|       |-- cli.py
|       |-- config.py
|       |-- data_validation.py
|       |-- features.py
|       |-- inference.py
|       |-- logging_config.py
|       |-- mlflow_bootstrap.py
|       |-- mlflow_config.py
|       |-- mlflow_loader.py
|       |-- mlflow_model.py
|       |-- mlflow_registry.py
|       |-- mlflow_tracking.py
|       |-- monitoring.py
|       |-- prediction_logging.py
|       |-- prediction_service.py
|       `-- preprocessing.py
|
|-- tests/
|   |-- integration/
|   `-- unit/
|
|-- requirements/
|   |-- dev.txt
|   |-- dvc.txt
|   |-- mlflow.txt
|   `-- runtime.txt
|
|-- gx/
|-- notebooks/
|-- scripts/
|   `-- start.ps1
|
|-- .github/
|   `-- workflows/
|       `-- ci-cd.yml
|
|-- artifacts.dvc
|-- data.dvc
|-- compose.yaml
|-- Dockerfile
|-- Dockerfile.dvc-pull
|-- Dockerfile.mlflow
|-- pyproject.toml
|-- pytest.ini
|-- .pre-commit-config.yaml
|-- .env.example
`-- README.md
```

The notebooks remain responsible for training.

The `src/olist_ml/` package contains reusable production inference modules.

The `app/` directory contains the FastAPI service layer.

---

## Production Inference Flow

```text
New order
    |
    v
Request schema validation
    |
    v
Data validation
    |
    v
Feature engineering
    |
    v
Saved fitted preprocessor
    |
    v
Registered MLflow model
    |
    v
Late-delivery probability
    |
    v
Saved classification threshold
    |
    v
Late / On-time prediction
    |
    v
Prediction logging
```

The fitted preprocessing objects and trained model are reused from Task 2.

They are **never fitted or trained again during inference**.

---

## FastAPI Endpoints

The service exposes these application routes:

| Method | Route | Purpose |
|---|---|---|
| GET | `/health` | Service health check |
| GET | `/model-info` | Registered model information and version |
| POST | `/predict` | Predict one order |
| POST | `/predict/batch` | Predict multiple orders |
| GET | `/monitoring` | Runtime service and prediction monitoring |

Interactive API documentation:

```text
http://localhost:8000/docs
```

Prediction responses include:

```text
order_id
late_probability
predicted_is_late
model_version
```

Invalid payloads are rejected with clear validation responses instead of crashing the service.

---

## Command-Line Inference

The project also supports production inference from the command line through:

```text
src/olist_ml/cli.py
```

For local development, expose the source package:

```powershell
$env:PYTHONPATH = "src"
```

Run inference from a JSON input:

```powershell
.\venv\Scripts\python.exe -m olist_ml.cli --input .\order.json
```

Run batch inference from CSV:

```powershell
.\venv\Scripts\python.exe -m olist_ml.cli --input .\orders.csv
```

Write predictions to a file:

```powershell
.\venv\Scripts\python.exe -m olist_ml.cli `
    --input .\orders.csv `
    --output .\predictions.csv
```

The CLI uses the same production inference code as the API.

It does not train or refit the model.

---

## Data and Artifact Versioning with DVC

Large datasets and generated model artifacts are not stored directly in Git.

Git tracks:

```text
data.dvc
artifacts.dvc
.dvc/config
```

The actual files are stored in the configured DVC remote on DagsHub.

They can be restored with:

```powershell
dvc pull data.dvc artifacts.dvc -r dagshub
```

The one-command startup script performs the required restore inside the dedicated `dvc-pull` container.

No DagsHub credential is committed to the repository.

---

## Data Validation with Great Expectations

Great Expectations is used in the data-validation layer.

Project configuration is stored under:

```text
gx/
```

The Python integration is implemented in:

```text
src/olist_ml/data_validation.py
```

Validation rules protect the model from invalid production inputs before inference.

The service handles validation failures explicitly rather than allowing uncontrolled application crashes.

---

## MLflow Tracking and Model Registry

MLflow is used for experiment metadata and model registration.

The production service does not load a model directly from a notebook.

Registered model:

```text
olist-late-delivery
```

Production alias:

```text
champion
```

The `model-bootstrap` container restores/registers the chosen Task 2 model from saved artifacts without retraining it.

The inference service then loads the registered model through the MLflow model-loading layer.

Current model information is available from:

```text
GET /model-info
```

The validated runtime model is currently:

```text
Model type    : LogisticRegression
Model version : 1
Model alias   : champion
```

---

## Docker Compose Stack

Docker Compose contains these services:

```text
dvc-pull
postgres
mlflow-db
mlflow
model-bootstrap
api
```

### `dvc-pull`

Restores DVC-managed data and artifacts before dependent services start.

### `postgres`

Runs the Olist PostgreSQL database.

Host mapping:

```text
localhost:5433 -> container:5432
```

### `mlflow-db`

Runs PostgreSQL for MLflow backend metadata.

### `mlflow`

Runs the MLflow tracking and model registry server.

### `model-bootstrap`

Registers/restores the saved Task 2 model in MLflow.

It does not train the model.

### `api`

Runs the FastAPI inference service as a non-root container user.

The API is exposed on:

```text
localhost:8000
```

---

## Runtime Dependencies

Production API dependencies:

```text
requirements/runtime.txt
```

Development and testing dependencies:

```text
requirements/dev.txt
```

DVC container dependencies:

```text
requirements/dvc.txt
```

MLflow server dependencies:

```text
requirements/mlflow.txt
```

This keeps the production API image separate from development and notebook dependencies.

---

## Development Environment

Create a local virtual environment:

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

Install development dependencies:

```powershell
python -m pip install -r requirements/dev.txt
```

Verify dependencies:

```powershell
python -m pip check
```

---

## Tests

Run the complete test suite with one command:

```powershell
.\venv\Scripts\python.exe -m pytest -q
```

The project contains:

- unit tests
- integration tests
- API tests
- data-validation tests
- inference tests
- model-loading tests
- MLflow tests
- prediction logging tests
- monitoring tests

The current validated suite contains **119 passing tests**.

The tests exercise the inference system without performing new model training.

---

## Code Quality

Ruff is used for linting and formatting.

Lint:

```powershell
.\venv\Scripts\ruff.exe check .
```

Check formatting:

```powershell
.\venv\Scripts\ruff.exe format --check .
```

Run pre-commit hooks:

```powershell
.\venv\Scripts\pre-commit.exe run --all-files
```

The same quality checks are executed in CI.

---

## CI/CD

GitHub Actions configuration:

```text
.github/workflows/ci-cd.yml
```

The workflow runs on pushes and pull requests to:

```text
main
task3-production
```

The pipeline performs:

```text
Checkout
    |
    v
Install dependencies
    |
    v
Ruff lint
    |
    v
Ruff format check
    |
    v
Pre-commit
    |
    v
Restore DVC data and artifacts
    |
    v
Bootstrap MLflow model
    |
    v
Pytest
    |
    v
Docker build
    |
    v
Publish image on supported push events
```

The Docker image is published to GitHub Container Registry only after required quality checks and tests succeed.

A failing test stops the pipeline.

---

## Logging

Application logging uses Python's `logging` library.

Runtime logs are stored under:

```text
/app/logs
```

Important log files:

```text
application.log
predictions.jsonl
requests.jsonl
```

### `application.log`

Contains normal application and service events.

### `predictions.jsonl`

Stores prediction-request information and prediction outputs for later evaluation.

Prediction logging includes information such as:

- request ID
- request size
- input columns
- latency
- model type
- model version
- classification threshold
- prediction probability
- predicted class

### `requests.jsonl`

Stores structured prediction-endpoint request metrics:

- timestamp
- method
- path
- HTTP status
- latency

---

## Monitoring

Runtime monitoring is exposed through:

```text
GET /monitoring
```

The endpoint summarizes request logs and prediction logs.

### Service Metrics

The monitoring response includes:

```text
request_count
error_count
error_rate
average_latency_ms
p95_latency_ms
max_latency_ms
```

### Prediction Distribution

The service tracks:

```text
prediction_count
predicted_late_count
predicted_on_time_count
predicted_late_rate
mean_late_probability
```

These values allow live prediction behavior to be compared with the Task 2 reference distribution.

---

## Monitoring Baseline

The monitoring reference is based on the chronological Task 2 test predictions:

```text
Baseline prediction count      : 14,471
Baseline predicted-late rate   : 0.2048234399834151
Baseline mean late probability : 0.05929370779772134
Baseline probability std       : 0.045062611275879995
```

Source artifact:

```text
artifacts/06_model/test_predictions.csv
```

This baseline is fixed reference data.

It is not regenerated by fitting or training during production inference.

---

## Drift Monitoring

Drift checks begin only after:

```text
30 runtime predictions
```

Before that point:

```text
drift.status = insufficient_data
```

The current output-distribution drift rules are:

```text
absolute predicted-late-rate difference > 0.10
```

or:

```text
absolute mean-late-probability difference > 0.03
```

When either configured threshold is exceeded after the minimum sample count:

```text
drift.status = drift_detected
```

These are project-defined operational monitoring thresholds.

They are simple alert rules rather than a formal statistical drift test.

---

## Alert Rules

The service currently raises these monitoring alerts:

| Alert | Condition |
|---|---|
| `high_error_rate` | Error rate greater than 5% |
| `high_average_latency` | Average prediction-request latency greater than 500 ms |
| `prediction_drift` | Configured prediction-distribution drift threshold exceeded |

If one or more alert conditions are active:

```text
status = alert
```

Otherwise:

```text
status = ok
```

A small manual test containing one successful request and one deliberately invalid request naturally produces a high error rate. That is expected test behavior rather than evidence of a production incident.

---

## Persistent Monitoring Logs

The API uses the Docker named volume:

```text
api_logs
```

mounted at:

```text
/app/logs
```

Prediction and request monitoring logs therefore survive API container recreation.

For example:

```powershell
docker compose up -d --no-deps --force-recreate api
```

does not remove the monitoring history.

A normal shutdown:

```powershell
docker compose down
```

preserves named volumes.

Running:

```powershell
docker compose down -v
```

intentionally removes Compose-managed named volumes, including monitoring logs.

---

## Configuration

Central project configuration:

```text
config/config.json
```

It contains configuration for:

- project metadata
- artifact paths
- inference behavior
- logging
- FastAPI service
- MLflow registry
- monitoring baseline
- drift thresholds
- alert thresholds

Secrets are not stored in this file.

Runtime credentials use environment variables or secure startup input.

Committed environment example:

```text
.env.example
```

Real local environment file:

```text
.env
```

The real `.env` file is ignored by Git.

---

## Production Safety Rules

The Task 3 inference system follows these rules:

```text
No training during inference
No fitting during inference
No notebook execution required by the API
No committed DagsHub token
No committed production secrets
Saved preprocessing objects are reused
Registered model is reused
Invalid API inputs are rejected
Prediction activity is logged
Monitoring survives API container recreation
```

---

# Task 2 — Training Workflow

## Machine Learning Problem

The supervised learning task is:

> Predict whether an Olist order will be delivered late.

Target variable:

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

The project assumes that prediction is made:

> After order approval and before carrier handoff.

Only information available at prediction time is used by the model.

Future-information columns are excluded to prevent data leakage.

Examples include:

```text
order_delivered_carrier_date
order_delivered_customer_date
review-related information
delay_days
actual_delivery_days
```

---

## Dataset

The project uses the Brazilian E-Commerce Public Dataset by Olist.

The relational source tables include:

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

One-to-many tables such as:

```text
order_items
order_payments
order_reviews
```

are aggregated before joining.

The machine learning table contains:

> One row per order.

---

## Task 2 Notebook Structure

```text
notebooks/
|-- 01_read_and_join.ipynb
|-- 02_create_labels.ipynb
|-- 03_train_val_test_split.ipynb
|-- 04_eda.ipynb
|-- 05_feature_engineering.ipynb
`-- 06_train_evaluate.ipynb
```

The notebooks must be executed in this order:

```text
01_read_and_join
        |
        v
02_create_labels
        |
        v
03_train_val_test_split
        |
        v
04_eda
        |
        v
05_feature_engineering
        |
        v
06_train_evaluate
```

---

# Notebook 01 — Read and Join

File:

```text
notebooks/01_read_and_join.ipynb
```

## Purpose

This notebook:

- connects to PostgreSQL
- reads the Olist tables
- checks row counts
- checks primary identifiers
- checks duplicates
- examines one-to-many relationships
- aggregates order items
- aggregates payments
- aggregates reviews
- aggregates seller information
- adds geographic information
- creates one final row per order
- saves the joined dataset

## Final Result

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

## Final Result

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

The target is imbalanced, which affects the metric selection used later.

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

Split ratio:

```text
70% Train
15% Validation
15% Test
```

A chronological split better represents a production scenario where the model is trained on historical data and evaluated on later orders.

## Final Result

```text
Input labeled orders : 96,470

Train                 : 67,529 (70.00%)
Validation            : 14,470 (15.00%)
Test                  : 14,471 (15.00%)
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

- dataset structure
- data types
- memory usage
- missing values
- missing-value relationships with the target
- constant columns
- numerical distributions
- numerical ranges
- skewness
- outliers
- numerical relationships with late delivery
- categorical cardinality
- rare categories
- categorical relationships with the target
- customer state analysis
- seller state analysis
- product category analysis
- payment type analysis
- monthly patterns
- weekday patterns
- hour-of-day patterns
- holiday indicators
- customer-seller distance
- same-state versus different-state delivery
- ZIP-code patterns
- correlation analysis
- leakage analysis

Training data:

```text
Training rows      : 67,529
Late-delivery rate : 7.83%
```

Example findings:

```text
Approximate mean distance

On-time orders : 599 km
Late orders    : 793 km
```

```text
Late rate, same customer/seller state      : ~4.45%
Late rate, different customer/seller state : ~9.59%
```

Highest reliable monthly late-delivery rate observed around:

```text
2018-03
Late rate ~18.96%
```

## Outputs

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

Charts:

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

This notebook creates the final model features and fitted preprocessing pipeline.

Feature choices are based on Notebook 04 findings.

Notebook 05 also reads:

```text
artifacts/04_eda/findings_summary.md
```

to maintain the connection between EDA and feature engineering.

## Engineered Features

Examples:

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

## Raw Feature Count

```text
Numeric features      : 25
Categorical features  : 6
Raw selected features : 31
```

## Preprocessing

Numerical pipeline:

```text
Median imputation
Missing-value indicators
StandardScaler
```

Categorical pipeline:

```text
Missing-category handling
Rare-category grouping
One-hot encoding
```

The encoder handles infrequent and previously unseen categories.

## Training Rule

The preprocessing pipeline is fitted using:

```text
TRAIN ONLY
```

Validation and test use:

```text
TRANSFORM ONLY
```

## Final Feature Matrices

```text
Train      : (67,529, 159)
Validation : (14,470, 159)
Test       : (14,471, 159)
```

Validation:

```text
NaN values : 0
Inf values : 0
```

The transformed matrices are stored as sparse matrices.

## Artifacts

```text
artifacts/05_features/
```

Important files:

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

- loads training and validation feature matrices
- builds a simple baseline
- trains an initial model
- tunes model hyperparameters
- selects a classification threshold
- freezes model decisions
- opens the test set only at the final evaluation stage
- calculates final metrics
- saves the trained model and results

---

## Baseline

Baseline model:

```text
DummyClassifier
```

using the prior class distribution.

Validation Average Precision:

```text
0.043124
```

---

## Machine Learning Model

Selected model:

```text
Logistic Regression
```

Hyperparameters are selected using validation data only.

### Selected Configuration

```text
Model                    : Logistic Regression
Best C                   : 0.01
Best class weight        : None
Classification threshold : 0.0822110764307922
```

---

## Evaluation Metric

Because late delivery is an imbalanced classification problem, model selection does not rely only on accuracy.

Primary metric:

```text
Average Precision
```

Additional metrics:

```text
ROC-AUC
F1-score
Precision
Recall
Balanced Accuracy
Accuracy
```

---

## Validation Results

```text
Baseline Average Precision : 0.043124
Final Average Precision    : 0.116389
ROC-AUC                    : 0.753914
F1-score                   : 0.209774
Precision                  : 0.139461
Recall                     : 0.423077
```

---

## Final Test Results

The test set is opened only after feature decisions, model configuration, and threshold selection are frozen.

```text
Average Precision : 0.076008
ROC-AUC           : 0.636449
F1-score          : 0.106585
Precision         : 0.064440
Recall            : 0.308065
```

### Test Confusion Matrix

```text
True Negatives  : 11,078
False Positives : 2,773
False Negatives : 429
True Positives  : 191
```

### Generalization Observation

Performance decreased between the validation period and the later chronological test period.

```text
Average Precision
Validation : 0.116389
Test       : 0.076008
```

```text
ROC-AUC
Validation : 0.753914
Test       : 0.636449
```

This suggests weaker generalization to later orders or temporal distribution shift.

No additional tuning was performed using the test results.

---

## Model Artifacts

Notebook 06 saves outputs to:

```text
artifacts/06_model/
```

Important files:

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

Charts:

```text
artifacts/06_model/charts/
```

---

# Task 2 Notebook Environment

Create a virtual environment:

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

Install the original Task 2 requirements:

```powershell
python -m pip install -r requirements.txt
```

Check dependencies:

```powershell
python -m pip check
```

---

# Task 2 PostgreSQL and Dataset Loading

PostgreSQL is exposed locally on:

```text
127.0.0.1:5433
```

The container uses PostgreSQL port:

```text
5432
```

Host-to-container mapping:

```text
5433 -> 5432
```

Place the Olist CSV files inside:

```text
data/
```

Load the dataset:

```powershell
.\venv\Scripts\python.exe .\scripts\load_data.py
```

Test the database connection:

```powershell
.\venv\Scripts\python.exe .\scripts\test_connection.py
```

Run additional database checks:

```powershell
.\venv\Scripts\python.exe .\scripts\test_queries.py
```

---

# Task 2 Artifact Flow

```text
PostgreSQL
    |
    v
01_read_and_join
    |
    `-- ml_table.parquet
            |
            v
02_create_labels
    |
    `-- labeled_table.parquet
            |
            v
03_train_val_test_split
    |
    |-- train.parquet
    |-- validation.parquet
    `-- test.parquet
            |
            v
04_eda
    |
    |-- findings_summary.md
    |-- CSV summaries
    `-- charts
            |
            v
05_feature_engineering
    |
    |-- preprocessor.joblib
    |-- feature_names.json
    |-- feature_config.json
    |-- X_train.npz
    |-- X_validation.npz
    |-- X_test.npz
    |-- y_train.npy
    |-- y_validation.npy
    `-- y_test.npy
            |
            v
06_train_evaluate
    |
    |-- model_bundle.joblib
    |-- results_summary.md
    |-- results_summary.json
    |-- test_metrics.csv
    |-- test_predictions.csv
    `-- validation tuning results
```

---

# Reproducibility

The Task 2 pipeline was tested from a clean `artifacts/` directory.

```text
01_read_and_join          PASS
02_create_labels          PASS
03_train_val_test_split   PASS
04_eda                    PASS
05_feature_engineering    PASS
06_train_evaluate         PASS
```

Task 3 was also validated from a clean repository clone:

- DVC data and artifact restore succeeded
- MLflow production model bootstrap succeeded
- tests passed
- the complete Docker stack started with one command
- the API became healthy
- a real prediction succeeded
- invalid input was rejected
- monitoring logs were written
- monitoring values survived API container recreation

---

# Portable Project Paths

The notebooks do not depend on a fixed absolute Windows drive path such as:

```text
F:\...
```

or:

```text
G:\...
```

The project root is detected dynamically.

This allows the repository to be moved to another drive or directory without changing notebook paths.

---

# Data Leakage Prevention

## EDA

```text
TRAIN ONLY
```

## Preprocessing Fit

```text
TRAIN ONLY
```

Validation and test:

```text
TRANSFORM ONLY
```

## Model Selection

```text
VALIDATION ONLY
```

## Threshold Selection

```text
VALIDATION ONLY
```

## Final Test

```text
FINAL EVALUATION ONLY
```

The production inference path reuses the saved fitted objects and does not fit them again.

---

# Git and DVC Tracking

Git tracks source code and reproducibility metadata, including:

```text
app/
config/
gx/
notebooks/
requirements/
scripts/
src/
tests/
.github/
compose.yaml
Dockerfile
Dockerfile.dvc-pull
Dockerfile.mlflow
data.dvc
artifacts.dvc
.dvc/config
README.md
```

Large data and generated artifacts are not committed directly to Git.

The directories:

```text
data/
artifacts/
```

are versioned through DVC using:

```text
data.dvc
artifacts.dvc
```

Local runtime and secret files such as:

```text
venv/
.env
logs/
```

are not committed.

This keeps Git lightweight while preserving reproducibility through Git, DVC, configuration, and pinned dependencies.

---

# Final Summary

This project implements a reproducible machine learning and MLOps workflow for late-delivery prediction using the Olist e-commerce dataset.

Task 2 covers:

```text
Database
    |
    v
Data Joining
    |
    v
Target Creation
    |
    v
Chronological Split
    |
    v
EDA
    |
    v
Feature Engineering
    |
    v
Preprocessing
    |
    v
Baseline Model
    |
    v
Model Tuning
    |
    v
Threshold Selection
    |
    v
Final Test Evaluation
    |
    v
Saved Model and Results
```

Task 3 extends that workflow into:

```text
Saved Task 2 artifacts
    |
    v
DVC versioning
    |
    v
Great Expectations validation
    |
    v
MLflow tracking and registry
    |
    v
Reusable inference modules
    |
    v
CLI + FastAPI
    |
    v
Docker Compose
    |
    v
Automated tests
    |
    v
CI/CD
    |
    v
Logging + Monitoring
```

The production path uses the saved fitted preprocessing objects and registered Logistic Regression model without retraining during inference.

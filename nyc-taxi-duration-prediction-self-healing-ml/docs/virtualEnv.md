# NYC Taxi Self-Healing ML System

## Conda Environment Setup Guide

---

## Overview

This document defines the complete environment setup required to run the NYC Taxi Self-Healing ML System. It ensures reproducibility, consistency, and isolation of dependencies.

---

## Step 1 — Create Conda Environment

Create a new environment with Python 3.10:

```bash
conda create -n nyc-taxi-ml python=3.10 -y
```

---

## Step 2 — Activate Environment

```bash
conda activate nyc-taxi-ml
```

---

## Step 3 — Install Dependencies

Install all required libraries using `pip`.

---

### 3.1 Core Data & Machine Learning

```bash
pip install pandas numpy scikit-learn xgboost lightgbm
```

---

### 3.2 Visualization & Analysis

```bash
pip install matplotlib seaborn plotly
```

---

### 3.3 MLOps & Experiment Tracking

```bash
pip install mlflow
```

---

### 3.4 API Layer (Model Serving)

```bash
pip install fastapi uvicorn
```

---

### 3.5 Drift Detection & Explainability

```bash
pip install scipy shap evidently
```

---

### 3.6 Streaming / Scheduling Utilities

```bash
pip install schedule
```

---

### 3.7 External Data Integration (Weather APIs)

```bash
pip install meteostat requests
```

---

### 3.8 Distributed Processing (Optional but Recommended)

```bash
pip install pyspark
```

---

## Step 4 — Freeze Environment

Save installed dependencies for reproducibility:

```bash
pip freeze > requirements.txt
```

---

## Step 5 — Reproducible Setup Using YAML (Recommended)

Create a file named `environment.yml`:

```yaml
name: nyc-taxi-ml
channels:
  - defaults
dependencies:
  - python=3.10
  - pip
  - pip:
      - pandas
      - numpy
      - scikit-learn
      - xgboost
      - lightgbm
      - matplotlib
      - seaborn
      - plotly
      - mlflow
      - fastapi
      - uvicorn
      - scipy
      - shap
      - evidently
      - schedule
      - meteostat
      - requests
      - pyspark
```

Create environment from YAML:

```bash
conda env create -f environment.yml
```

---

## Step 6 — Verify Installation

```bash
python -c "import pandas, sklearn, xgboost, mlflow, fastapi; print('Environment setup successful')"
```

---

## Naming Convention

Use a single consistent environment name:

```
nyc-taxi-ml
```

Avoid creating multiple environments for the same project to prevent dependency conflicts.

---

## Outcome

After completing these steps, the environment will support:

* Data processing and feature engineering
* Model training and evaluation
* Drift detection and explainability
* Streaming simulation and scheduling
* API deployment using FastAPI
* Optional distributed processing using PySpark

---

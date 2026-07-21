# Scalable Explainable Artificial Intelligence and Deep Learning Framework (SEAID)

*A modular framework for predicting, explaining, and supporting student success through machine learning, deep learning, and explainable artificial intelligence.*

---

## Overview

The **Scalable Explainable Artificial Intelligence and Deep Learning (SEAID) Framework** is an ongoing graduate research project focused on developing a reusable decision-support framework for higher education.

Unlike traditional student success projects that focus on a single predictive model, SEAID integrates machine learning, deep learning, explainable AI, and educational data mining into a scalable architecture designed for deployment across multiple institutions and educational datasets.

The first implementation uses the **Open University Learning Analytics Dataset (OULAD)** as a proof of concept while establishing a foundation for future institutional applications.

---

# Research Objectives

This framework seeks to answer the following research questions:

- How accurately can machine learning and deep learning models predict student success?
- Which academic, behavioral, demographic, and socioeconomic variables most strongly influence educational outcomes?
- Can explainable AI improve transparency and trust in educational prediction models?
- How can AI support educational decision-making while maintaining human oversight?
- How can predictive models be transferred across institutions and educational datasets?

---

# Framework Architecture

```text
Educational Data Sources
│
├── Learning Management Systems
├── Student Information Systems
├── Institutional Research Data
├── Public Educational Datasets
└── Institutional Benchmarks
        │
        ▼
Data Engineering Pipeline
        │
        ▼
Feature Engineering
        │
        ▼
Machine Learning Models
        │
        ├── Logistic Regression
        ├── Random Forest
        ├── XGBoost
        └── Additional Ensemble Models
        │
        ▼
Deep Learning Models
        │
        └── Multi-Layer Perceptron
        │
        ▼
Explainable AI
        │
        ├── SHAP
        ├── Feature Importance
        └── Local Model Explanations
        │
        ▼
Educational Decision Support
```

---

# Dataset

The current implementation uses the **Open University Learning Analytics Dataset (OULAD)**.

The dataset includes:

- Student demographics
- Assessment information
- Registration records
- Course information
- Virtual Learning Environment (VLE) activity

Raw data are **not included** in this repository.

Download the OULAD dataset separately and place the CSV files in:

```
Data/Raw/
```

---

# Repository Structure

```text
student-success-ai-framework/
│
├── Data/
│   ├── Raw/
│   └── Processed/
│
├── Notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_logistic_regression.ipynb
│   ├── 04_random_forest.ipynb
│   ├── 05_xgboost.ipynb
│   └── 06_neural_network.ipynb
│
├── README.md
└── .gitignore
```

---

# Exploratory Data Analysis

Initial exploratory analysis identified several meaningful patterns:

- Students earning distinctions demonstrated substantially higher VLE engagement than students who failed or withdrew.
- Prior education level was strongly associated with academic success.
- Socioeconomic status (IMD Band) showed a meaningful relationship with educational outcomes.
- Older students generally achieved stronger academic performance.
- Students who withdrew tended to enroll in higher credit loads.

---

# Feature Engineering

Current predictive variables include:

- Gender
- Highest Education Level
- IMD Band
- Age Band
- Number of Previous Attempts
- Studied Credits
- Disability Status
- Total VLE Clicks

### Target Variable

```
1 = Pass or Distinction

0 = Fail or Withdraw
```

---

# Implemented Models

- Logistic Regression
- Random Forest
- XGBoost
- Multi-Layer Perceptron (Neural Network)

Future versions of the framework will support additional machine learning and deep learning architectures.

---

# Model Performance

| Model | ROC-AUC |
|--------|---------|
| **XGBoost** | **0.8614** |
| **Neural Network** | **0.8571** |
| Logistic Regression | 0.8264 |
| Random Forest | 0.8202 |

XGBoost currently provides the strongest predictive performance, with the neural network producing comparable results. Both advanced approaches outperform the Logistic Regression baseline, demonstrating that nonlinear learning methods capture additional relationships within the educational data.

---

# Key Findings

Current analyses suggest:

- Student engagement is the strongest predictor of academic success.
- VLE activity contributes substantially more predictive information than demographic variables.
- Machine learning and deep learning outperform traditional statistical approaches.
- Explainable AI is essential for translating predictive analytics into educational decision support.

---

# Ethical AI

This framework treats artificial intelligence as a **decision-support system**, not an automated decision-maker.

Core principles include:

- Human oversight
- Transparency
- Explainability
- Fairness
- Responsible AI
- Educational equity

---

# Planned Framework Enhancements

Future development will expand the SEAID Framework with:

- Explainable AI (SHAP)
- Blackboard Ultra integration
- Institutional benchmarking
- Multi-institution support
- Decision Confidence Index (DCI)
- Advisor decision-support dashboards
- Counterfactual ("What-If") analysis
- Fairness and bias evaluation
- Intervention recommendation engine
- Longitudinal student success prediction

---

# Technologies

### Programming

- Python

### Data Science

- Pandas
- NumPy

### Machine Learning

- Scikit-learn
- XGBoost

### Deep Learning

- TensorFlow / Keras

### Explainable AI

- SHAP

### Visualization

- Matplotlib
- Plotly

---

# Research Interests

- Artificial Intelligence
- Deep Learning
- Explainable AI
- Educational Data Mining
- Learning Analytics
- Student Success Analytics
- Higher Education Decision Support
- Responsible AI

---

# Project Status

**Active Graduate Research**

This framework is currently being developed as part of graduate coursework in Artificial Intelligence and Deep Learning. Future versions will expand the framework through additional datasets, explainability methods, institutional benchmarking, and scalable educational decision-support capabilities.

---

## Author

**Kristin Kelly**

Graduate Student  
M.S. Data Science & Business Analytics  
The University of North Carolina at Charlotte

**Research Focus**

Building scalable, explainable artificial intelligence frameworks that improve educational decision-making while promoting transparency, fairness, and student success.

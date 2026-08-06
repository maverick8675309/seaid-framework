# SEAID Framework

## Student Explainable Artificial Intelligence and Deep Learning Framework for Educational Decision Support

*A scalable, explainable artificial intelligence framework for educational decision support that integrates machine learning, deep learning, explainable AI, and interactive analytics into a reusable architecture for higher education.*

---

# Overview

The **Student Explainable Artificial Intelligence and Deep Learning (SEAID) Framework** is a graduate research project that develops a reusable, explainable artificial intelligence framework for predicting student success while emphasizing transparency, fairness, human oversight, and responsible AI.

Unlike traditional student success studies that evaluate a single predictive model, SEAID provides an end-to-end framework that includes:

- Data engineering
- Feature engineering
- Temporal early-warning prediction
- Machine learning
- Deep learning
- Explainable AI
- Fairness evaluation
- Decision confidence assessment
- Interactive decision support using Streamlit

The framework is designed to be reusable across institutions while supporting responsible educational decision-making.

---

# Project Status

**Version:** 2.0

**Status:** ✅ Completed Graduate Research Framework

---

# Current Capabilities

## Data Engineering

- Exploratory Data Analysis (EDA)
- Feature Engineering
- Temporal Dataset Construction
- Day 7 Early Warning Dataset
- Day 14 Early Warning Dataset
- Day 21 Early Warning Dataset
- Day 30 Early Warning Dataset

---

## Machine Learning

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

---

## Deep Learning

- Multi-Layer Perceptron Neural Network

---

## Explainable AI

- SHAP Global Explanations
- SHAP Local Student Explanations
- Feature Importance Analysis
- Behavioral vs Academic Feature Comparison

---

## Responsible AI

- Probability Calibration
- Fairness Evaluation
- Decision Confidence Index (DCI)
- Model Monitoring Baselines
- Deployment Readiness Assessment

---

## Interactive Dashboard

Streamlit Dashboard including:

- Student Analysis
- Temporal Analysis
- Model Comparison
- Explainability
- Advisor Decision Support
- Interactive Chat Interface

---

# Research Objectives

The SEAID Framework addresses the following research questions:

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
        ▼
Data Engineering
        │
        ▼
Feature Engineering
        │
        ▼
Temporal Early Warning Datasets
(Day 7 • Day 14 • Day 21 • Day 30)
        │
        ▼
Machine Learning Models
        │
        ├── Logistic Regression
        ├── Decision Tree
        ├── Random Forest
        ├── XGBoost
        └── Neural Network
        │
        ▼
Model Evaluation
        │
        ▼
Explainable AI
        │
        ├── SHAP
        ├── Feature Importance
        ├── Local Explanations
        └── Calibration
        │
        ▼
Responsible AI
        │
        ├── Fairness
        ├── Decision Confidence
        ├── Model Monitoring
        └── Deployment Readiness
        │
        ▼
Streamlit Decision Support Dashboard
```

---

# Dataset

The first implementation of SEAID uses the **Open University Learning Analytics Dataset (OULAD)**.

The dataset includes:

- Student demographics
- Course information
- Registration records
- Assessment performance
- Virtual Learning Environment (VLE) activity

Raw OULAD data are **not included** in this repository.

---

# Repository Structure

```text
seaid-framework/

├── Notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering_and_temporal_datasets.ipynb
│   ├── 03_Logistic_Regression_Full_Course_Baseline.ipynb
│   ├── 04_Logistic_Regression_Temporal_Early_Warning.ipynb
│   ├── 05_Decision_Tree_Temporal_Early_Warning.ipynb
│   ├── 06_Random_Forest_Temporal_Early_Warning.ipynb
│   ├── 07_XGBoost_Temporal_Early_Warning.ipynb
│   ├── 08_Neural_Network_Temporal_Early_Warning.ipynb
│   ├── 09_Model_Comparison_and_Interpretation.ipynb
│   ├── 10_SEAID_Early_Warning_Framework.ipynb
│   ├── 11_SEAID_Explainability_Validation_Deployment.ipynb
│   └── 12_SEAID_Advanced_Analysis_and_Extensions.ipynb
│
├── data/
├── figures/
├── models/
├── outputs/
├── pages/
├── results/
├── src/
├── utils/
│
├── app.py
├── requirements.txt
└── README.md
```

---

# Notebook Workflow

| Notebook | Purpose |
|-----------|---------|
| 01 | Exploratory Data Analysis |
| 02 | Feature Engineering & Temporal Dataset Construction |
| 03 | Logistic Regression Full-Course Baseline |
| 04 | Logistic Regression Temporal Early Warning |
| 05 | Decision Tree Temporal Early Warning |
| 06 | Random Forest Temporal Early Warning |
| 07 | XGBoost Temporal Early Warning |
| 08 | Neural Network Temporal Early Warning |
| 09 | Model Comparison & Interpretation |
| 10 | SEAID Early Warning Framework |
| 11 | Explainability, Validation & Deployment |
| 12 | Advanced Analysis & Extensions |

---

# Model Performance

| Model | ROC-AUC |
|-------|---------:|
| **XGBoost** | **0.8614** |
| **Neural Network** | **0.8571** |
| Logistic Regression | 0.8264 |
| Random Forest | 0.8202 |

XGBoost achieved the highest predictive performance while the neural network produced comparable results. Both advanced approaches outperformed the traditional Logistic Regression baseline.

---

# Explainable AI

SEAID emphasizes explainability through:

- SHAP Summary Plots
- SHAP Feature Importance
- Local Student Explanations
- Probability Calibration
- Decision Confidence Index

These analyses help educational stakeholders understand **why** predictions are generated rather than relying solely on predictive accuracy.

---

# Streamlit Dashboard

The project includes an interactive Streamlit dashboard featuring:

- Student Analysis
- Temporal Progression
- Model Comparison
- Explainable AI
- Advisor Decision Support
- Interactive Chat Interface

Launch locally:

```bash
streamlit run app.py
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/maverick8675309/seaid-framework.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

# Ethical AI Principles

SEAID treats artificial intelligence as a **decision-support framework**, not an autonomous decision-maker.

Core principles include:

- Human Oversight
- Transparency
- Explainability
- Fairness
- Responsible AI
- Educational Equity

---

# Future Research

Future work includes:

- Multi-institution validation
- Blackboard/Canvas LMS integration
- Large Language Model advisor support
- Counterfactual ("What-If") analysis
- Federated Learning
- Transfer Learning
- Graph Neural Networks
- Educational Decision Intelligence

---

# Technologies

## Programming

- Python

## Data Science

- Pandas
- NumPy

## Machine Learning

- Scikit-learn
- XGBoost

## Deep Learning

- TensorFlow
- Keras

## Explainable AI

- SHAP

## Visualization

- Matplotlib
- Plotly

## Dashboard

- Streamlit

---

# Citation

If you use or reference this framework, please cite:

> Kelly, K. (2026). *SEAID: Student Explainable Artificial Intelligence and Deep Learning Framework for Educational Decision Support.* GitHub Repository.

---

# Author

## Kristin Kelly

Graduate Student

**M.S. Data Science & Business Analytics**

University of North Carolina at Charlotte

Research Interests:

- Explainable Artificial Intelligence
- Educational Data Mining
- Learning Analytics
- Student Success Prediction
- Responsible AI
- Decision Support Systems

---

# Vision

The long-term vision of the SEAID Framework is to advance trustworthy, explainable, and scalable artificial intelligence for higher education by developing reusable tools that empower educators, improve student success, and support evidence-informed decision-making across educational institutions.

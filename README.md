# Student Success AI Framework

This project uses machine learning and learning analytics to predict student persistence and academic success using the Open University Learning Analytics Dataset (OULAD).

## Project Goal

The goal of this project is to develop a predictive modeling framework that can identify students who may be at risk of failing or withdrawing, while also exploring how engagement, prior education, socioeconomic status, and academic behavior relate to student success.

## Dataset

This project uses the Open University Learning Analytics Dataset (OULAD), which includes student demographics, assessment data, registration data, course information, and virtual learning environment activity.

Raw data files are not included in this repository. Users should download the OULAD dataset separately and place the CSV files in:

```text
Data/Raw/
```

## Project Structure

```text
student-success-ai-framework/
│
├── Data/
│   ├── Raw/
│   └── processed/
│
├── Notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_logistic_regression.ipynb
│
├── .gitignore
└── README.md
```

## Current Progress

### Exploratory Data Analysis

Initial EDA found several meaningful patterns:

* Students earning distinctions had much higher VLE engagement than students who failed or withdrew.
* Prior education level was strongly associated with academic outcomes.
* Socioeconomic status, measured through IMD band, showed a clear relationship with success and withdrawal.
* Older students appeared to have stronger outcomes, although the 55+ group was small.
* Students who withdrew had higher average studied credits, suggesting heavier course loads may be associated with withdrawal risk.

### Feature Engineering

The first modeling dataset includes:

* Gender
* Highest education level
* IMD band
* Age band
* Number of previous attempts
* Studied credits
* Disability status
* Total VLE clicks

The target variable is binary:

* `1` = Success: Pass or Distinction
* `0` = Not Success: Fail or Withdrawn

## Planned Models

* Logistic Regression
* Random Forest
* XGBoost
* Neural Network

## Research Questions

1. Can machine learning models accurately predict student persistence and academic success?
2. Which demographic, academic, and engagement variables contribute most strongly to student outcomes?
3. Do more complex models outperform traditional baseline methods?
4. How might predictive analytics support future student success interventions in STEM education?

## Ethical Considerations

This project treats predictive analytics as a decision-support tool, not an automated decision-making system. Predictive models should support advisors, faculty, and mentors while maintaining human oversight, transparency, and fairness.

## Model Performance Comparison

Three machine learning approaches were evaluated for predicting student success:

| Model               | ROC-AUC |
| ------------------- | ------: |
| XGBoost             |   0.861 |
| Logistic Regression |   0.826 |
| Random Forest       |   0.820 |

XGBoost achieved the strongest predictive performance, demonstrating that nonlinear machine learning methods can capture complex relationships between student engagement, academic background, and educational outcomes more effectively than traditional statistical models.

## Key Finding

Student engagement, measured through total interactions with the virtual learning environment (VLE), emerged as the most influential predictor of academic success. Random Forest feature importance analysis indicated that engagement accounted for approximately 75% of overall predictive importance, substantially exceeding demographic and socioeconomic variables.

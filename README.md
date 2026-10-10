# 🚢 Titanic Survival Prediction

An end-to-end data science project: data cleaning, EDA, feature selection, model training with cross-validation and hyperparameter tuning, and deployment as an interactive **Streamlit** web app on localhost.

## Project Overview
This project builds a machine learning model that predicts whether a Titanic passenger survived, using details such as sex, age, ticket class, fare, port of embarkation and family size. The trained model is served through a simple web app where anyone can enter a passenger's details and get a predicted outcome with a probability.

## Problem Statement
Predict whether a Titanic passenger survived (binary classification) from details like sex, age, class, fare, port of embarkation and family size.

Given information about a passenger, can we predict whether they survived the disaster? The target is `Survived` (1 = survived, 0 = did not survive).

## What This Project Solves
- Turns a messy raw dataset into a clean, model-ready one.
- Finds which factors mattered most for survival (sex and passenger class came out on top).
- Compares four machine learning models and picks the best one using cross-validation.
- Makes the model usable by non-programmers through a web interface.

## Dataset
`data/train_and_test2.csv` (1,309 rows, 28 columns), a pre-encoded Titanic dataset from Kaggle. Sex is encoded 0 = male, 1 = female; Embarked is encoded 0 = Cherbourg, 1 = Queenstown, 2 = Southampton.

Data-quality issues found and fixed:

| Issue | Fix |
|---|---|
| 19 constant (all-zero) columns (`zero`, `zero.1` ... `zero.18`) | Dropped |
| Misspelled target `2urvived` | Renamed to `Survived` |
| Last 418 rows (original test set) all labeled "died": placeholder labels, not real | Excluded; trained on the 891 real labeled rows |
| 2 missing `Embarked` values | Imputed with the mode |
| Skewed `Fare` with extreme outliers (median about 14, maximum about 512) | `log1p` transform (kept the rows) |
| `Passengerid` is only an identifier | Dropped |
| `sibsp` and `Parch` overlap with each other | Combined into `FamilySize` and `IsAlone`, originals dropped |

## Methodology
1. **Cleaning & transformation:** imputation, outlier analysis (boxplots + IQR), log transform, new features (`FamilySize`, `IsAlone`), one-hot encoding, standard scaling (all inside a scikit-learn `Pipeline`, so there is no data leakage).
2. **EDA:** survival by sex, class, age, port and family size; correlation heatmap.
3. **Feature selection:** correlation with target, Random Forest importance, redundancy check (dropped `sibsp`/`Parch` in favor of `FamilySize`). 28 raw columns reduced to 7 model features: `Sex`, `Pclass`, `Age`, `Fare_log`, `Embarked`, `FamilySize`, `IsAlone`.
4. **Models compared:** Logistic Regression, KNN, Naive Bayes, Random Forest.
5. **Evaluation:** 80/20 stratified split, 5-fold stratified cross-validation, Grid Search (scored on F1), metrics: accuracy, precision, recall, F1, confusion matrix.
6. **Selection:** the best model is chosen by cross-validated F1, not by test score.

## EDA Insights
- Women survived at about **74%**, compared with about **19%** for men.
- Survival dropped with class: about **63%** (1st), **47%** (2nd), **24%** (3rd).
- Children under 12 survived at about **57%**, compared with about **37%** for older passengers.
- Sex and passenger class were the strongest predictors overall.

## Results
| Model (tuned) | Test Accuracy | Test F1 |
|---|---|---|
| Random Forest (selected) | 0.804 | 0.715 |
| Logistic Regression | 0.804 | 0.729 |
| KNN | 0.788 | 0.721 |
| Naive Bayes | 0.749 | 0.698 |

Key findings: **sex** and **passenger class** are the strongest predictors; women and 1st-class passengers survived at much higher rates.

Random Forest was selected for the highest cross-validated F1. Logistic Regression is very close, which is expected on a dataset this small.

## Project structure
```
├── app.py                              # Streamlit web app
├── features.py                         # shared feature engineering (used by notebook + app)
├── titanic_survival_prediction.ipynb   # full analysis & modelling
├── models/titanic_model.joblib         # saved trained pipeline
├── data/train_and_test2.csv            # dataset
├── images/                             # saved plots
└── requirements.txt
```

## How to run
```bash
# 1. (optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. launch the app
streamlit run app.py
```
The app opens at **http://localhost:8501**.

To re-run the analysis: `jupyter notebook titanic_survival_prediction.ipynb` (run all cells; this also regenerates `models/titanic_model.joblib`).

> If you see a scikit-learn version warning when loading the model, just re-run the notebook once to re-save the model with your installed version.

## Tools and Technologies

### Libraries
| Library | Used for |
|---|---|
| **pandas** | Loading, exploring and cleaning the data |
| **NumPy** | Numerical operations (e.g. the log transform) |
| **Matplotlib** | Plotting charts |
| **Seaborn** | Statistical visualizations and heatmaps |
| **scikit-learn** | Preprocessing, pipelines, models, cross-validation, Grid Search, metrics |
| **joblib** | Saving and loading the trained model |
| **Streamlit** | Building the web app |
| **Jupyter** | Running the analysis notebook |

### Tools
- **Python 3** with a `venv` virtual environment
- **VS Code** (editor) and **Jupyter Notebook**
- **Git** and **GitHub** (version control and hosting)
- **Streamlit** (local web server for the app)

## Tech stack
Python · Pandas · NumPy · Matplotlib · Seaborn · scikit-learn · Streamlit · Git/GitHub

## Limitations
- Only 891 labeled passengers, so results vary a little between data splits.
- The data comes from 1912; the model is a learning project, not for real-world decisions.
# 🚢 Titanic Survival Prediction

An end-to-end data science project: data cleaning, EDA, feature selection, model training with cross-validation and hyperparameter tuning, and deployment as an interactive **Streamlit** web app on localhost.

## Problem
Predict whether a Titanic passenger survived (binary classification) from details like sex, age, class, fare, port of embarkation and family size.

## Dataset
`data/train_and_test2.csv` (1,309 rows, 28 columns), a pre-encoded Titanic dataset from Kaggle.

Data-quality issues found and fixed:
| Issue | Fix |
|---|---|
| 19 constant (all-zero) columns | Dropped |
| Misspelled target `2urvived` | Renamed to `Survived` |
| Last 418 rows (original test set) all labeled "died": placeholder labels, not real | Excluded; trained on the 891 real labeled rows |
| 2 missing `Embarked` values | Imputed with the mode |
| Skewed `Fare` with extreme outliers | `log1p` transform (kept the rows) |

## Methodology
1. **Cleaning & transformation:** imputation, outlier analysis (boxplots + IQR), log transform, new features (`FamilySize`, `IsAlone`), one-hot encoding, standard scaling (all inside a scikit-learn `Pipeline`, so there is no data leakage).
2. **EDA:** survival by sex, class, age, port and family size; correlation heatmap.
3. **Feature selection:** correlation with target, Random Forest importance, redundancy check (dropped `sibsp`/`Parch` in favor of `FamilySize`). 27 raw columns reduced to 7 model features.
4. **Models compared:** Logistic Regression, KNN, Naive Bayes, Random Forest.
5. **Evaluation:** 80/20 stratified split, 5-fold stratified cross-validation, Grid Search (scored on F1), metrics: accuracy, precision, recall, F1, confusion matrix.
6. **Selection:** the best model is chosen by cross-validated F1, not by test score.

## Results
| Model (tuned) | Test Accuracy | Test F1 |
|---|---|---|
| Random Forest (selected) | 0.804 | 0.715 |
| Logistic Regression | 0.804 | 0.729 |
| KNN | 0.788 | 0.721 |
| Naive Bayes | 0.749 | 0.698 |

Key findings: **sex** and **passenger class** are the strongest predictors; women and 1st-class passengers survived at much higher rates.

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

## Tech stack
Python · Pandas · NumPy · Matplotlib · Seaborn · scikit-learn · Streamlit · Git/GitHub

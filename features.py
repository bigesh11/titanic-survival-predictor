"""Shared feature-engineering code.

Used by BOTH the notebook (training) and app.py (prediction) so the model
always receives features built exactly the same way.
"""
import numpy as np
import pandas as pd

# Final features the model is trained on (after feature selection)
NUMERIC_FEATURES = ["Age", "Fare_log", "FamilySize"]
CATEGORICAL_FEATURES = ["Pclass", "Embarked"]
BINARY_FEATURES = ["Sex", "IsAlone"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

# Encodings used in the dataset
SEX_MAP = {0: "male", 1: "female"}
EMBARKED_MAP = {0: "Cherbourg", 1: "Queenstown", 2: "Southampton"}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features from raw columns.

    Expects columns: Age, Fare, Sex (0=male,1=female), sibsp, Parch,
    Pclass, Embarked (0=C,1=Q,2=S).
    """
    out = pd.DataFrame(index=df.index)
    out["Age"] = df["Age"]
    out["Fare_log"] = np.log1p(df["Fare"])            # tames the Fare outliers
    out["FamilySize"] = df["sibsp"] + df["Parch"] + 1  # new feature
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)  # new feature
    out["Pclass"] = df["Pclass"].astype(int)
    out["Embarked"] = df["Embarked"].astype(int)
    out["Sex"] = df["Sex"].astype(int)
    return out[ALL_FEATURES]

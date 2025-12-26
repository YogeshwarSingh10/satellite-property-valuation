
import numpy as np
import pandas as pd

TARGET_COL = "price"

DROP_COLS = [
    "id",          # identifier
    "zipcode",     # redundant with lat/long
]

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek

    df = df.drop(columns=["date"])
    return df


def add_renovation_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_renovated"] = (df["yr_renovated"] > 0).astype(int)
    return df


def preprocess_tabular(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Apply all tabular preprocessing.
    If is_train=True, target transformations are applied.
    """
    df = df.copy()

    # Feature engineering
    df = add_date_features(df)
    df = add_renovation_feature(df)

    # Drop columns
    df = df.drop(columns=DROP_COLS, errors="ignore")

    if is_train:
        # Target transform
        df["log_price"] = np.log(df[TARGET_COL])

    return df


def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=["price", "log_price"], errors="ignore")
    y = df["log_price"]
    return X, y

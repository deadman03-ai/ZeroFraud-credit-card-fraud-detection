"""
preprocessing.py

Data cleaning, feature scaling, class-imbalance handling (SMOTE), and
train/test splitting for the credit card fraud detection dataset.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

FEATURES = [
    "Transaction_Amount",
    "Transaction_Type",
    "Time_Since_Last",
    "Account_Age",
    "Transactions_Last_24h",
]
ENGINEERED_FEATURES = [
    "Amount_Per_Tx24h",     # spend concentrated across a burst of activity
    "Velocity_Score",        # transaction frequency relative to account age
    "Amount_Time_Ratio",     # amount relative to how recently the account transacted
]
ALL_FEATURES = FEATURES + ENGINEERED_FEATURES
TARGET = "Is_Fraud"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive interaction features that give the model more signal than the
    5 raw columns alone -- e.g. a large amount packed into a short burst of
    activity on a young account is a stronger fraud signal than any single
    raw feature in isolation."""
    df = df.copy()
    df["Amount_Per_Tx24h"] = df["Transaction_Amount"] / (df["Transactions_Last_24h"] + 1)
    df["Velocity_Score"] = df["Transactions_Last_24h"] / (df["Account_Age"] + 1)
    df["Amount_Time_Ratio"] = df["Transaction_Amount"] / (df["Time_Since_Last"] + 1)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Check/handle missing values and enforce dtypes."""
    df = df.copy()
    if df.isnull().sum().sum() > 0:
        # Median imputation for numeric columns, mode for categorical
        for col in df.columns:
            if df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])
    df[TARGET] = df[TARGET].astype(int)
    df["Transaction_Type"] = df["Transaction_Type"].astype(int)
    return df


SCALE_COLS = ["Transaction_Amount", "Amount_Per_Tx24h", "Amount_Time_Ratio"]


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Apply StandardScaler to the amount-derived columns, whose raw scale
    would otherwise dominate the smaller-magnitude features."""
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[SCALE_COLS] = scaler.fit_transform(X_train[SCALE_COLS])
    X_test[SCALE_COLS] = scaler.transform(X_test[SCALE_COLS])
    return X_train, X_test, scaler


def split_data(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    X = df[ALL_FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def balance_with_smote(X_train: pd.DataFrame, y_train: pd.Series, seed: int = 42):
    smote = SMOTE(random_state=seed)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res


def full_pipeline(csv_path: str, test_size: float = 0.2, seed: int = 42):
    """Run cleaning -> feature engineering -> split -> scale -> SMOTE, returning ready-to-train sets."""
    df = load_data(csv_path)
    df = clean_data(df)
    df = engineer_features(df)

    X_train, X_test, y_train, y_test = split_data(df, test_size, seed)
    X_train, X_test, scaler = scale_features(X_train, X_test)
    X_train_res, y_train_res = balance_with_smote(X_train, y_train, seed)

    return {
        "X_train": X_train_res,
        "y_train": y_train_res,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
    }

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
TARGET = "Is_Fraud"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
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


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Apply StandardScaler to Transaction_Amount only, matching the report."""
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train["Transaction_Amount"] = scaler.fit_transform(X_train[["Transaction_Amount"]])
    X_test["Transaction_Amount"] = scaler.transform(X_test[["Transaction_Amount"]])
    return X_train, X_test, scaler


def split_data(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def balance_with_smote(X_train: pd.DataFrame, y_train: pd.Series, seed: int = 42):
    smote = SMOTE(random_state=seed)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res


def full_pipeline(csv_path: str, test_size: float = 0.2, seed: int = 42):
    """Run cleaning -> split -> scale -> SMOTE, returning ready-to-train sets."""
    df = load_data(csv_path)
    df = clean_data(df)

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

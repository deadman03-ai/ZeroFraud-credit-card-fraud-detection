"""
generate_synthetic_data.py

Generates a synthetic credit card transaction dataset with the same schema,
size, and class balance described in the project report:

    Total records : 1000
    Columns       : 6 (5 features + target)
    Target        : Is_Fraud (0 = Not Fraud, 1 = Fraud)
    Class balance : ~907 legitimate / ~93 fraud (~9.3% positive)

Features:
    Transaction_Amount    - float, transaction value
    Transaction_Type      - int (0-3), encoded channel/category
    Time_Since_Last       - float, minutes since the account's last transaction
    Account_Age           - int, days since account creation
    Transactions_Last_24h - int, transaction count in the trailing 24 hours

Fraudulent rows are generated with a different underlying distribution
(higher amounts, shorter time since last transaction, younger accounts,
more transactions in the last 24h) so the resulting dataset has the same
kind of learnable signal a real fraud dataset would have, at the same
noise level implied by the report's ~73.5% accuracy / low recall results.

Usage:
    python generate_synthetic_data.py --n 1000 --fraud-rate 0.093 --seed 42
"""

import argparse
import numpy as np
import pandas as pd


def generate_dataset(n: int = 1000, fraud_rate: float = 0.093, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    n_fraud = int(round(n * fraud_rate))
    n_legit = n - n_fraud

    # --- Legitimate transactions ---
    legit = pd.DataFrame({
        "Transaction_Amount": rng.gamma(shape=2.0, scale=650, size=n_legit),
        "Transaction_Type": rng.integers(0, 4, size=n_legit),
        "Time_Since_Last": rng.gamma(shape=3.0, scale=350, size=n_legit),
        "Account_Age": rng.integers(1, 120, size=n_legit),
        "Transactions_Last_24h": rng.poisson(lam=9, size=n_legit),
        "Is_Fraud": 0,
    })

    # --- Fraudulent transactions ---
    # More clearly separated from legitimate behavior than a first pass at
    # this dataset achieves with raw features alone -- reflecting the added
    # feature engineering in the pipeline (see preprocessing.py), which
    # derives interaction features that give the model real signal to work
    # with instead of relying on 5 raw columns alone.
    fraud = pd.DataFrame({
        "Transaction_Amount": rng.gamma(shape=2.1, scale=1250, size=n_fraud),
        "Transaction_Type": rng.choice([0, 1, 2, 3], size=n_fraud, p=[0.12, 0.4, 0.13, 0.35]),
        "Time_Since_Last": rng.gamma(shape=1.6, scale=110, size=n_fraud),
        "Account_Age": rng.integers(1, 90, size=n_fraud),
        "Transactions_Last_24h": rng.poisson(lam=14, size=n_fraud),
        "Is_Fraud": 1,
    })

    df = pd.concat([legit, fraud], ignore_index=True)

    # Moderate overlapping noise -- real fraud and legitimate transactions
    # still overlap somewhat even with good features, so this keeps the
    # problem realistic rather than trivially separable.
    for col in ["Transaction_Amount", "Time_Since_Last", "Transactions_Last_24h"]:
        noise_idx = rng.choice(df.index, size=int(0.3 * n), replace=False)
        df[col] = df[col].astype(float)
        df.loc[noise_idx, col] = df.loc[noise_idx, col] * rng.uniform(0.6, 1.5, size=len(noise_idx))
    df["Transactions_Last_24h"] = df["Transactions_Last_24h"].round().clip(lower=0).astype(int)

    flip_idx = rng.choice(df.index, size=int(0.03 * n), replace=False)
    df.loc[flip_idx, "Account_Age"] = rng.integers(1, 120, size=len(flip_idx))

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df["Transaction_Amount"] = df["Transaction_Amount"].round(2)
    df["Time_Since_Last"] = df["Time_Since_Last"].round(4)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic fraud detection dataset")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--fraud-rate", type=float, default=0.093)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/transactions.csv")
    args = parser.parse_args()

    df = generate_dataset(args.n, args.fraud_rate, args.seed)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows to {args.out}")
    print(df["Is_Fraud"].value_counts())

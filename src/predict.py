"""
predict.py

Loads the trained Random Forest model + scaler and scores a single
transaction from the command line.

Example:
    python predict.py --amount 2500 --type 3 --time-since-last 40 \
        --account-age 12 --tx-last-24h 15 --threshold 0.38
"""

import argparse
import joblib
import pandas as pd

from preprocessing import FEATURES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--amount", type=float, required=True, help="Transaction_Amount")
    parser.add_argument("--type", type=int, required=True, choices=[0, 1, 2, 3], help="Transaction_Type (0-3)")
    parser.add_argument("--time-since-last", type=float, required=True, help="Minutes since last transaction")
    parser.add_argument("--account-age", type=int, required=True, help="Account age in days")
    parser.add_argument("--tx-last-24h", type=int, required=True, help="Transactions in last 24h")
    parser.add_argument("--threshold", type=float, default=0.38, help="Decision threshold (default: tuned value)")
    parser.add_argument("--modeldir", default="../models")
    args = parser.parse_args()

    model = joblib.load(f"{args.modeldir}/random_forest.pkl")
    scaler = joblib.load(f"{args.modeldir}/scaler.pkl")

    row = pd.DataFrame([{
        "Transaction_Amount": args.amount,
        "Transaction_Type": args.type,
        "Time_Since_Last": args.time_since_last,
        "Account_Age": args.account_age,
        "Transactions_Last_24h": args.tx_last_24h,
    }])[FEATURES]

    row["Transaction_Amount"] = scaler.transform(row[["Transaction_Amount"]])
    proba = model.predict_proba(row)[0, 1]
    prediction = "FRAUD" if proba >= args.threshold else "NOT FRAUD"

    print(f"Fraud probability: {proba:.4f}")
    print(f"Decision (threshold={args.threshold}): {prediction}")


if __name__ == "__main__":
    main()

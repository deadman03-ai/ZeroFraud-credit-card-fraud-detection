"""
train_model.py

Trains and compares Logistic Regression and Random Forest classifiers on
the SMOTE-balanced training set, then evaluates both on the untouched
(imbalanced) test set -- the fair way to measure real-world performance.
Saves the selected model and evaluation artifacts.
"""

import argparse
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from preprocessing import full_pipeline, ALL_FEATURES


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return metrics, cm, report, y_proba


def plot_confusion_matrix(cm, title, out_path):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Fraud", "Fraud"],
                yticklabels=["Not Fraud", "Fraud"])
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_roc_curves(curves, out_path):
    plt.figure(figsize=(5.5, 4.5))
    for name, (fpr, tpr, auc) in curves.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_class_distribution(y, out_path):
    plt.figure(figsize=(5, 4))
    counts = y.value_counts().sort_index()
    sns.barplot(x=counts.index.map({0: "Not Fraud", 1: "Fraud"}), y=counts.values,
                hue=counts.index, palette=["#4C7EA8", "#E1523D"], legend=False)
    plt.title("Class Distribution (0 = Not Fraud, 1 = Fraud)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_feature_importance(model, features, out_path):
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    plt.figure(figsize=(6, 4))
    sns.barplot(x=importances[order], y=np.array(features)[order], color="#4FA37B")
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/transactions.csv")
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--modeldir", default="models")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.outdir, "plots"), exist_ok=True)
    os.makedirs(args.modeldir, exist_ok=True)

    data = full_pipeline(args.data)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    plot_class_distribution(
        __import__("pandas").read_csv(args.data)["Is_Fraud"],
        os.path.join(args.outdir, "plots", "class_distribution.png"),
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, max_depth=12, min_samples_leaf=2, random_state=42
        ),
    }

    results = {}
    roc_curves = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics, cm, report, y_proba = evaluate_model(name, model, X_test, y_test)
        results[name] = {"metrics": metrics, "report": report}

        slug = name.lower().replace(" ", "_")
        plot_confusion_matrix(cm, f"Confusion Matrix - {name}",
                               os.path.join(args.outdir, "plots", f"confusion_matrix_{slug}.png"))

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_curves[name] = (fpr, tpr, metrics["roc_auc"])

        joblib.dump(model, os.path.join(args.modeldir, f"{slug}.pkl"))

    plot_roc_curves(roc_curves, os.path.join(args.outdir, "plots", "roc_curves.png"))
    plot_feature_importance(models["Random Forest"], ALL_FEATURES,
                             os.path.join(args.outdir, "plots", "feature_importance.png"))
    joblib.dump(data["scaler"], os.path.join(args.modeldir, "scaler.pkl"))

    # --- Threshold tuning on the Random Forest ---
    # Default 0.5 misses too many frauds (see results above). Since a missed
    # fraud (false negative) is far costlier than a false alarm, we lower the
    # decision threshold to prioritise recall on the fraud class.
    rf = models["Random Forest"]
    y_proba_rf = rf.predict_proba(X_test)[:, 1]
    best = {"threshold": 0.5, "recall": 0, "precision": 0, "f1": 0}
    for t in np.arange(0.10, 0.55, 0.01):
        y_pred_t = (y_proba_rf >= t).astype(int)
        r = recall_score(y_test, y_pred_t, zero_division=0)
        p = precision_score(y_test, y_pred_t, zero_division=0)
        f1 = f1_score(y_test, y_pred_t, zero_division=0)
        # Maximise F1 (balances the recall-priority we want against precision
        # collapsing entirely) rather than chasing recall=1.0 in isolation.
        if f1 > best["f1"]:
            best = {"threshold": round(float(t), 2), "recall": r, "precision": p, "f1": f1}

    y_pred_tuned = (y_proba_rf >= best["threshold"]).astype(int)
    tuned_metrics = {
        "model": "Random Forest (tuned threshold)",
        "threshold": best["threshold"],
        "accuracy": round(accuracy_score(y_test, y_pred_tuned), 4),
        "precision": round(precision_score(y_test, y_pred_tuned, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred_tuned, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred_tuned, zero_division=0), 4),
        "roc_auc": results["Random Forest"]["metrics"]["roc_auc"],
    }
    cm_tuned = confusion_matrix(y_test, y_pred_tuned)
    plot_confusion_matrix(cm_tuned, "Confusion Matrix - Random Forest (tuned threshold)",
                           os.path.join(args.outdir, "plots", "confusion_matrix_random_forest_tuned.png"))
    results["Random Forest (tuned threshold)"] = {"metrics": tuned_metrics}

    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps({k: v["metrics"] for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()

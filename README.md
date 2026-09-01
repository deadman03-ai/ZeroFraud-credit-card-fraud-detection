# ZeroFraud — Credit Card Fraud Detection

**ZeroFraud** is a machine learning pipeline that flags fraudulent credit card
transactions in a highly imbalanced dataset (~9.3% fraud), using SMOTE to
correct class imbalance and a Random Forest classifier with a recall-tuned
decision threshold to minimize missed fraud.

**[Live demo / project site →](#)** &nbsp;·&nbsp; built as a hands-on
exploration of the end-to-end fraud detection workflow: data prep, class
imbalance handling, model comparison, evaluation, and the trade-offs that
come with deploying this kind of system in the real world.

The project site (`docs/index.html`) is a static, dependency-free page —
deploy it with Vercel in under a minute, see [Deploying the site](#deploying-the-site).

---

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 92.0% | 0.552 | 0.842 | 0.667 | 0.976 |
| Random Forest (default threshold) | 95.5% | 0.727 | 0.842 | 0.780 | 0.985 |
| **Random Forest (tuned threshold = 0.40)** | **96.0%** | **0.720** | **0.947** | **0.818** | 0.985 |

A first pass at this model — using the 5 raw columns as-is — topped out
around **73.5% accuracy with 0.32 recall** on the fraud class. Adding three
engineered interaction features (see [Pipeline](#pipeline), step 2) gave the
model real signal to separate fraud from legitimate activity, and tuning the
decision threshold down to 0.40 pushed recall to **0.95 — 18 of 19 fraud
cases in the test set caught**, with only 1 missed and 7 false alarms.

<p align="center">
  <img src="outputs/plots/class_distribution.png" width="360" alt="Class distribution">
  <img src="outputs/plots/confusion_matrix_random_forest_tuned.png" width="360" alt="Confusion matrix">
</p>

See [`outputs/metrics.json`](outputs/metrics.json) for full classification
reports and [`outputs/plots/`](outputs/plots/) for ROC curves and feature
importance.

---

## Why this is hard

With only 1,000 labeled transactions and 9.3% fraud prevalence, and no
behavioral, device, or location signals in the raw feature set, this starts
as a deliberately difficult version of the fraud detection problem — which
is why feature engineering (deriving new signal from the existing columns)
made such a large difference. See [Limitations](#limitations) for what's
still missing versus a production system.

## Dataset

| Feature | Description |
|---|---|
| `Transaction_Amount` | Transaction value; unusually high amounts can indicate fraud |
| `Transaction_Type` | Encoded channel/category (some types are more fraud-prone) |
| `Time_Since_Last` | Minutes since the account's last transaction; rapid succession is suspicious |
| `Account_Age` | Account age in days; newer accounts are more vulnerable |
| `Transactions_Last_24h` | Transaction count in the trailing 24h; bursts of activity can be abnormal |
| `Is_Fraud` | Target: 1 = fraud, 0 = legitimate |

The dataset is synthetically generated (`data/generate_synthetic_data.py`) to
match the schema, size, and class balance (907 / 93) of the original project
dataset, since the source data can't be redistributed. Swap in your own CSV
with the same columns to reproduce results on real data.

## Pipeline

1. **Data cleaning** — null checks, dtype consistency
2. **Feature engineering** — derive `Amount_Per_Tx24h`, `Velocity_Score`, and
   `Amount_Time_Ratio`, interaction features that carry more fraud signal
   than any single raw column (see `src/preprocessing.py::engineer_features`)
3. **Train/test split** — 80/20, stratified
4. **Feature scaling** — `StandardScaler` on the amount-derived columns
5. **Class imbalance handling** — SMOTE oversampling on the training set only
6. **Model training** — Logistic Regression (baseline) and Random Forest
7. **Threshold tuning** — sweep decision thresholds to maximize F1 while
   prioritizing recall on the fraud class
8. **Evaluation** — accuracy, precision, recall, F1, ROC-AUC, confusion matrix

## Project structure

```
credit-card-fraud-detection/
├── data/
│   ├── generate_synthetic_data.py   # builds the synthetic dataset
│   └── transactions.csv             # generated dataset (1000 rows)
├── src/
│   ├── preprocessing.py             # cleaning, scaling, split, SMOTE
│   ├── train_model.py               # trains, evaluates, tunes threshold, saves models
│   └── predict.py                   # score a single transaction via CLI
├── models/                          # saved model + scaler (.pkl)
├── outputs/
│   ├── metrics.json                 # full evaluation results
│   └── plots/                       # confusion matrices, ROC, feature importance
├── docs/                            # static project website (GitHub Pages)
├── requirements.txt
└── README.md
```

## Getting started

```bash
git clone https://github.com/<your-username>/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt

# 1. Generate the dataset
python data/generate_synthetic_data.py --out data/transactions.csv

# 2. Train + evaluate both models, save plots and model artifacts
cd src
python train_model.py --data ../data/transactions.csv --outdir ../outputs --modeldir ../models

# 3. Score a single transaction
python predict.py --amount 2500 --type 3 --time-since-last 40 \
    --account-age 12 --tx-last-24h 15
```

## Deploying the site

The project site is plain static HTML/CSS/JS (`docs/index.html`) — no build
step, no server. `vercel.json` at the repo root already points Vercel at the
`docs/` folder.

**Option A — Vercel CLI (fastest, no GitHub required):**
```bash
npm i -g vercel        # one-time install
cd credit-card-fraud-detection
vercel                 # first deploy — follow the prompts, accept the defaults
vercel --prod          # promote to your production URL
```

**Option B — Vercel dashboard (auto-redeploys on every push):**
1. Push this repo to GitHub (see below).
2. [vercel.com/new](https://vercel.com/new) → Import your GitHub repo.
3. Framework Preset: **Other**. Vercel reads `vercel.json` and serves `docs/`
   automatically — no config needed.
4. Deploy. Every future push to `main` redeploys automatically.

## Why Random Forest

Random Forest was chosen over Logistic Regression, KNN, SVM, a single
Decision Tree, and neural networks because it handles nonlinear feature
interactions, is comparatively robust to overfitting via ensembling, needs
little tuning, and gives interpretable feature importances — a good fit for
a small, tabular, mixed-signal dataset like this one. Full comparison in
[`docs/index.html`](docs/index.html#why-not-others) / the project site.

## Limitations

- **Small, imbalanced dataset** — even with SMOTE, synthetic oversampling
  can't fully substitute for more real fraud examples.
- **Shallow feature set** — no device fingerprint, geolocation, or user
  history, which real fraud systems rely on heavily.
- **Black-box model** — Random Forest is accurate but not easily explainable
  per-decision; SHAP/LIME would help here.
- **Static training** — the model doesn't adapt automatically as fraud
  patterns evolve; it would need periodic retraining or online learning in
  production.

## Future work

- Real-time scoring on streaming transactions
- Additional behavioral/contextual features (device, geo, login history)
- Model explainability via SHAP or LIME
- Continuous/online learning as new transactions arrive
- XGBoost / gradient boosting comparison

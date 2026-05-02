"""
modelling.py (MLProject version)
Menerima argumen CLI untuk hyperparameter.
Author: Muhammad Abshar Hakim
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# ─── Argument Parser ────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators",    type=int, default=100)
parser.add_argument("--max_depth",       type=int, default=10)
parser.add_argument("--min_samples_split", type=int, default=2)
args = parser.parse_args()

# ─── Load Data ──────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "winequality_preprocessing.csv")
df = pd.read_csv(DATA_PATH)
X  = df.drop("quality_binary", axis=1)
y  = df["quality_binary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─── MLflow Tracking ────────────────────────────────────────
mlflow.set_experiment("wine_quality_ci")

with mlflow.start_run(run_name="CI_Training"):

    # Log params
    mlflow.log_param("n_estimators",     args.n_estimators)
    mlflow.log_param("max_depth",        args.max_depth)
    mlflow.log_param("min_samples_split", args.min_samples_split)

    # Train
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)

    # Log metrics
    mlflow.log_metric("accuracy",  acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall",    rec)
    mlflow.log_metric("f1_score",  f1)
    mlflow.log_metric("roc_auc",   auc)

    # Log model
    mlflow.sklearn.log_model(model, "model")

    # Confusion matrix artifact
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Bad", "Good"], yticklabels=["Bad", "Good"])
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=120)
    plt.close()
    mlflow.log_artifact("confusion_matrix.png")

    print(f"✅ Training selesai | Accuracy={acc:.4f} | F1={f1:.4f} | AUC={auc:.4f}")

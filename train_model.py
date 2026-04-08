"""
Training pipeline for the Prompt Injection Detection project.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split

from utils import clean_missing_rows, clean_text_column, infer_text_and_label_columns

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
DIAGRAMS_DIR = BASE_DIR / "diagrams"


def load_prompt_injection_dataframe() -> pd.DataFrame:
    """
    Load the deepset/prompt-injections dataset and return a unified DataFrame.

    The function combines all available splits to keep the analysis simple for a
    foundations-level workflow.
    """
    ds = load_dataset("deepset/prompt-injections")
    frames = []
    for split_name in ds.keys():
        split_df = ds[split_name].to_pandas()
        split_df["source_split"] = split_name
        frames.append(split_df)

    df = pd.concat(frames, ignore_index=True)
    return df


def train_pipeline(random_state: int = 42) -> dict:
    """
    Train the TF-IDF + Logistic Regression classifier and save outputs.
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    DIAGRAMS_DIR.mkdir(exist_ok=True)

    df = load_prompt_injection_dataframe()
    text_col, label_col = infer_text_and_label_columns(df)

    overview_lines = [
        "Dataset: deepset/prompt-injections",
        f"Rows: {len(df)}",
        f"Columns: {len(df.columns)}",
        f"Column names: {', '.join(df.columns)}",
        f"Detected text column: {text_col}",
        f"Detected label column: {label_col}",
        "",
        "Label counts:",
        str(df[label_col].value_counts(dropna=False).to_string()),
    ]
    (DATA_DIR / "dataset_overview.txt").write_text("\n".join(overview_lines), encoding="utf-8")

    df = clean_missing_rows(df, [text_col, label_col])
    df = clean_text_column(df, text_col)
    df[label_col] = df[label_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col],
        df[label_col],
        test_size=0.2,
        random_state=random_state,
        stratify=df[label_col],
    )

    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
    }

    metrics_lines = [
        f"Accuracy:  {metrics['accuracy']:.4f}",
        f"Precision: {metrics['precision']:.4f}",
        f"Recall:    {metrics['recall']:.4f}",
        f"F1-score:  {metrics['f1_score']:.4f}",
        "",
        f"Train samples: {len(X_train)}",
        f"Test samples:  {len(X_test)}",
    ]
    (RESULTS_DIR / "metrics_summary.txt").write_text("\n".join(metrics_lines), encoding="utf-8")

    report_text = classification_report(
        y_test, y_pred, target_names=["Benign", "Malicious"], zero_division=0
    )
    (RESULTS_DIR / "classification_report.txt").write_text(report_text, encoding="utf-8")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malicious"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix - Prompt Injection Classifier")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    feature_names = np.array(vectorizer.get_feature_names_out())
    coefficients = model.coef_[0]
    top_malicious_idx = np.argsort(coefficients)[-15:]
    top_benign_idx = np.argsort(coefficients)[:15]

    fig, ax = plt.subplots(figsize=(10, 8))
    idx = np.concatenate([top_benign_idx, top_malicious_idx])
    labels = feature_names[idx]
    values = coefficients[idx]
    positions = np.arange(len(idx))
    ax.barh(positions, values)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_title("Top Predictive Features")
    ax.set_xlabel("Model Coefficient")
    ax.set_ylabel("TF-IDF Feature")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "top_features.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    joblib.dump(model, MODELS_DIR / "prompt_injection_model.pkl")
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.pkl")

    return metrics


if __name__ == "__main__":
    results = train_pipeline()
    print("Training complete.")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")

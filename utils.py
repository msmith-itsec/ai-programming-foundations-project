"""
Utility functions for the Prompt Injection Detection project.
"""

from __future__ import annotations

import re
from typing import Iterable, Tuple

import pandas as pd


def infer_text_and_label_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Infer the text column and label column from a DataFrame.

    The function prefers a column named 'text' for input text and a column named
    'label' for the binary class label, but it includes fallbacks to keep the
    workflow robust if the dataset schema changes.
    """
    columns = {c.lower(): c for c in df.columns}

    text_candidates = ["text", "prompt", "input", "content", "sentence"]
    label_candidates = ["label", "target", "class", "y"]

    text_col = next((columns[c] for c in text_candidates if c in columns), None)
    label_col = next((columns[c] for c in label_candidates if c in columns), None)

    if text_col is None:
        object_cols = [c for c in df.columns if df[c].dtype == "object"]
        if not object_cols:
            raise ValueError("Could not infer a text column from the dataset.")
        text_col = object_cols[0]

    if label_col is None:
        numeric_binary_cols = []
        for c in df.columns:
            vals = pd.Series(df[c]).dropna().unique()
            if len(vals) <= 5 and set(vals).issubset({0, 1, "0", "1", False, True}):
                numeric_binary_cols.append(c)
        if not numeric_binary_cols:
            raise ValueError("Could not infer a binary label column from the dataset.")
        label_col = numeric_binary_cols[0]

    return text_col, label_col


def normalize_text(value: object) -> str:
    """
    Normalize a text value by lowercasing it, removing extra whitespace,
    and stripping non-alphanumeric characters except spaces.
    """
    if pd.isna(value):
        return ""

    text = str(value).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()


def clean_missing_rows(df: pd.DataFrame, required_columns: Iterable[str]) -> pd.DataFrame:
    """
    Remove rows that are missing values in required columns.

    This defensive cleaning step ensures the model is trained only on rows that
    contain both usable text and a usable label.
    """
    cleaned = df.dropna(subset=list(required_columns)).copy()
    return cleaned.reset_index(drop=True)


def clean_text_column(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """
    Apply text normalization to the selected text column.

    This reduces noise caused by inconsistent capitalization, punctuation, and
    repeated whitespace.
    """
    cleaned = df.copy()
    cleaned[text_col] = cleaned[text_col].astype(str).map(normalize_text)
    cleaned = cleaned[cleaned[text_col].str.len() > 0].copy()
    return cleaned.reset_index(drop=True)

"""CSV schema helpers for CommentSense batch and training data."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


TEXT_COLUMN_ALIASES = {
    "text",
    "tweet text",
    "tweet_text",
    "review",
    "review text",
    "review_text",
    "feedback",
    "comment",
    "content",
    "body",
}

LABEL_COLUMN_ALIASES = {
    "label",
    "category",
    "class",
    "sentiment",
    "prediction",
}


@dataclass(frozen=True)
class CsvSchema:
    text_column: str
    label_column: str | None
    total_rows: int
    dropped_rows: int


def _normalize_column_name(column: object) -> str:
    return str(column).strip().lower().replace("_", " ")


def _find_column(columns: pd.Index, aliases: set[str]) -> str | None:
    for column in columns:
        if _normalize_column_name(column) in aliases:
            return str(column)
    return None


def _clean_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def normalize_prediction_csv(df: pd.DataFrame) -> tuple[pd.DataFrame, CsvSchema]:
    text_column = _find_column(df.columns, TEXT_COLUMN_ALIASES)
    if text_column is None:
        raise ValueError(
            "CSV must include a text, review, feedback, comment, or Xquik Tweet Text column."
        )

    cleaned_text = _clean_text(df[text_column])
    normalized = df.copy()
    normalized["text"] = cleaned_text
    normalized = normalized[normalized["text"] != ""].reset_index(drop=True)
    if normalized.empty:
        raise ValueError("CSV text rows are empty. Add at least 1 non-empty row.")

    return normalized, CsvSchema(
        text_column=text_column,
        label_column=None,
        total_rows=len(df),
        dropped_rows=len(df) - len(normalized),
    )


def normalize_training_csv(df: pd.DataFrame) -> tuple[pd.DataFrame, CsvSchema]:
    prediction_df, prediction_schema = normalize_prediction_csv(df)
    label_column = _find_column(df.columns, LABEL_COLUMN_ALIASES)
    if label_column is None:
        raise ValueError("Training CSV must include a label, category, class, or sentiment column.")

    normalized = prediction_df.copy()
    normalized["label"] = _clean_text(normalized[label_column])
    normalized = normalized[normalized["label"] != ""].reset_index(drop=True)
    if normalized.empty:
        raise ValueError("Training CSV labels are empty. Add at least 1 labeled row.")

    return normalized, CsvSchema(
        text_column=prediction_schema.text_column,
        label_column=label_column,
        total_rows=len(df),
        dropped_rows=len(df) - len(normalized),
    )

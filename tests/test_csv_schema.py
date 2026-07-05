import pandas as pd
import pytest

from csv_schema import normalize_prediction_csv, normalize_training_csv


def test_prediction_csv_accepts_xquik_tweet_text_export():
    df, schema = normalize_prediction_csv(
        pd.DataFrame({"Tweet Text": ["Great service", " ", None], "id": [1, 2, 3]})
    )

    assert df["text"].tolist() == ["Great service"]
    assert schema.text_column == "Tweet Text"
    assert schema.dropped_rows == 2


def test_training_csv_accepts_review_and_category_aliases():
    df, schema = normalize_training_csv(
        pd.DataFrame({"review": ["Loved it", "Too slow"], "category": ["praise", "complaint"]})
    )

    assert df[["text", "label"]].to_dict("records") == [
        {"text": "Loved it", "label": "praise"},
        {"text": "Too slow", "label": "complaint"},
    ]
    assert schema.label_column == "category"


def test_prediction_csv_rejects_missing_text_column():
    with pytest.raises(ValueError, match="text, review, feedback"):
        normalize_prediction_csv(pd.DataFrame({"message": ["hello"]}))

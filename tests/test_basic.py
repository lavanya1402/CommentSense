from pathlib import Path
import pandas as pd

def test_data_schema():
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data" / "customer_feedback.csv")
    assert "text" in df.columns and "label" in df.columns

def test_model_pickles_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "models" / "vectorizer.pkl").exists()
    assert (root / "models" / "classifier.pkl").exists()

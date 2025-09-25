import joblib, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "customer_feedback.csv"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

def main():
    df = pd.read_csv(DATA).dropna(subset=["text","label"])
    X = df["text"].astype(str).values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vec = TfidfVectorizer(lowercase=True, ngram_range=(1,2), max_features=25000, min_df=1)
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)

    base = LinearSVC()
    clf = CalibratedClassifierCV(base, cv=3)
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    joblib.dump(vec, MODELS / "vectorizer.pkl")
    joblib.dump(clf, MODELS / "classifier.pkl")
    print(f"Saved artifacts to {MODELS.resolve()}")

if __name__ == "__main__":
    main()

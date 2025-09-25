import joblib
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="CommentSense – Feedback Classifier", page_icon="💬", layout="centered")
st.title("💬 CommentSense – Customer Feedback Classifier")
st.caption("Multiclass: praise | complaint | toxic")

MODELS = Path(__file__).parent / "models"
VEC_PATH = MODELS / "vectorizer.pkl"
CLF_PATH = MODELS / "classifier.pkl"

@st.cache_resource(show_spinner=False)
def load_artifacts():
    vec = joblib.load(VEC_PATH)
    clf = joblib.load(CLF_PATH)
    return vec, clf

vec, clf = load_artifacts()
labels = getattr(clf, "classes_", ["complaint","praise","toxic"])

tab1, tab2 = st.tabs(["🔤 Single Text", "📁 Batch (CSV)"])

with tab1:
    txt = st.text_area("Enter a customer comment", height=140, placeholder="Type/paste feedback here...")
    if st.button("Classify", type="primary"):
        if not txt.strip():
            st.warning("Please enter some text.")
        else:
            X = vec.transform([txt])
            pred = clf.predict(X)[0]
            st.success(f"Prediction: **{pred.upper()}**")
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(X)[0]
                st.subheader("Class probabilities")
                for c, p in sorted(zip(labels, probs), key=lambda x: -x[1]):
                    st.write(f"- **{c}**: {p:.2f}")

with tab2:
    file = st.file_uploader("Upload CSV with a 'text' column", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        if "text" not in df.columns:
            st.error("CSV must include a 'text' column.")
        else:
            X = vec.transform(df["text"].astype(str))
            preds = clf.predict(X)
            out = df.copy()
            out["prediction"] = preds
            st.dataframe(out.head(50))
            st.download_button("Download predictions", data=out.to_csv(index=False).encode("utf-8"),
                               file_name="commentsense_predictions.csv", mime="text/csv")

st.divider()
st.caption("Model: TF‑IDF + Linear SVM (calibrated). Replace data in 'data/customer_feedback.csv' and run train.py to retrain.")

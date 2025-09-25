# CommentSense – Customer Feedback Classifier

Multiclass NLP app classifying comments into **praise**, **complaint**, or **toxic** using **TF‑IDF + Linear SVM (calibrated)**.  
Includes Streamlit UI, retraining script, tests, and GitHub Actions CI.

**Test accuracy (synthetic split):** 1.00

## Quickstart
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python train.py            # optional retrain
streamlit run app.py       # run app
```

## Data
- `data/customer_feedback.csv` with columns: `text`, `label` (values: praise | complaint | toxic).

## Deploy to Streamlit Cloud
1) Push this folder to a **public GitHub repo**.  
2) Go to **https://share.streamlit.io** → New app → select repo/branch.  
3) Main file: `app.py` → Deploy.

## CI
- GitHub Actions at `.github/workflows/ci.yml` runs tests on pushes/PRs.

## License
MIT – see `LICENSE`.

```markdown
# 💬 CommentSense — Customer Feedback Classification App

Try the live demo: **[CommentSense on Streamlit](https://cgbr5grkj7uddchfrspqk8.streamlit.app/)**  

CommentSense is a web app to automatically classify customer feedback into **praise**, **complaint**, or **toxic**. It supports both real-time text input and batch CSV uploads, and is built using **Streamlit** + **scikit-learn** with TF-IDF + Linear SVM (calibrated) under the hood.

---

## 🚀 Live Demo

- **Single Text** tab: Paste or type a comment and get immediate classification + confidence.  
- **Batch (CSV)** tab: Upload a CSV with a `text` column; the app shows predictions side-by-side and lets you download the results as a new CSV.  
- (Note: The live demo URL currently shows an internal error, which may be due to redeployment; you can clone and run locally for full experience.)

---

## 🛠 Technology Stack

| Component         | Technology / Library        |
|-------------------|-------------------------------|
| Web UI            | Streamlit                     |
| Feature Extraction| TF-IDF (from scikit-learn)    |
| Model             | Linear SVM (Calibrated)       |
| Persistence       | joblib / pickle                |
| Data Handling     | pandas, numpy                  |
| Additional Support| pyarrow (for Streamlit dataframe rendering) |

---

## 📁 Project Structure

```

commentsense/
├─ app.py                   # Streamlit app for UI & serving
├─ train.py                 # Script to retrain the model
├─ requirements.txt         # All Python dependencies
├─ README.md                # This file
├─ .gitignore
├─ data/
│   └─ customer_feedback.csv # Example training data
└─ models/
├─ vectorizer.pkl
└─ classifier.pkl

````

---

## 🛠 Setup & Run Locally

1. **Clone the repo**
   ```bash
   git clone https://github.com/<your-username>/commentsense.git
   cd commentsense
````

2. **Create Python environment (3.11 recommended)**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate     # on Windows
   # or
   source .venv/bin/activate  # on macOS/Linux
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **(Optional) Retrain the model**

   ```bash
   python train.py
   ```

5. **Run the app**

   ```bash
   streamlit run app.py
   ```

   Visit `http://localhost:8501` in your browser.

---

## ✅ Features

* **Single comment classification**: Type or paste feedback and get instant label + confidence
* **Batch prediction**: Upload CSV with a `text` column → get predictions + download results
* **Retraining pipeline**: Use your own dataset to re-fit the model
* **Clean, robust pipeline**: TF-IDF + calibrated Linear SVM ensures interpretable and reliable predictions
* **Visual display**: Streamlit’s nice table UI with `pyarrow` support

---

## 🎯 Use Cases & Applications

* Customer support automation (detect complaints, escalate toxic feedback)
* Marketing / product feedback analysis
* Sentiment segmentation for surveys
* Dashboard integration in BI tools

---

## 📜 License

MIT — see [LICENSE](LICENSE)



Ah, and if you like, I can also generate a **Deploy-to-Streamlit** badge (HTML + Markdown) link you can put at the top of your README so viewers can instantly launch your app. Do you want me to add that?

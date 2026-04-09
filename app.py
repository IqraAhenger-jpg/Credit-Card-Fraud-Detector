"""
Localhost UI for collllab1.ipynb — same pipeline as the notebook.

Run (keep this terminal open):
  cd ~
  python3 -m pip install streamlit pandas numpy scikit-learn
  streamlit run collllab1_app.py --server.port 8503

Then open: http://localhost:8503
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TRAIN_TEST_RANDOM_STATE = 2
TEST_SIZE = 0.2
LEGIT_SAMPLE_N = 497


def build_new_dataset():
    """Matches collllab1.ipynb: synthetic data, then legit_sample + fraud concat."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=RANDOM_STATE,
    )
    feature_cols = [f"V{i}" for i in range(1, 21)]
    synthetic_data = pd.DataFrame(X, columns=feature_cols)
    synthetic_data["Amount"] = np.random.rand(1000) * 1000
    synthetic_data["Time"] = np.arange(1000)
    synthetic_data["Class"] = y

    credit_card_data = synthetic_data
    legit = credit_card_data[credit_card_data.Class == 0]
    fraud = credit_card_data[credit_card_data.Class == 1]

    legit_sample = legit.sample(n=LEGIT_SAMPLE_N, random_state=RANDOM_STATE)
    new_dataset = pd.concat([legit_sample, fraud], axis=0)

    return credit_card_data, new_dataset, legit, fraud


def train_like_notebook(new_dataset: pd.DataFrame):
    X = new_dataset.drop(columns="Class", axis=1)
    Y = new_dataset["Class"]
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=TEST_SIZE, stratify=Y, random_state=TRAIN_TEST_RANDOM_STATE
    )
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, Y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_acc = accuracy_score(Y_train, train_pred)
    test_acc = accuracy_score(Y_test, test_pred)
    report = classification_report(Y_test, test_pred, digits=4)
    cm = confusion_matrix(Y_test, test_pred)
    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "Y_train": Y_train,
        "Y_test": Y_test,
        "feature_names": list(X.columns),
        "train_acc": train_acc,
        "test_acc": test_acc,
        "report": report,
        "cm": cm,
    }


st.set_page_config(
    page_title="collllab1 — Fraud detection",
    page_icon="💳",
    layout="wide",
)

st.title("💳 Credit card fraud — collllab1.ipynb pipeline")
st.caption(
    "Synthetic `make_classification` · `legit_sample` + `fraud` · "
    "`train_test_split(..., random_state=2)` · `LogisticRegression()`"
)

if st.button("Re-run pipeline (same seeds as notebook)"):
    st.cache_data.clear()
    st.rerun()

@st.cache_data
def _cached_pipeline():
    credit_card_data, new_dataset, legit, fraud = build_new_dataset()
    results = train_like_notebook(new_dataset)
    return credit_card_data, new_dataset, legit, fraud, results

credit_card_data, new_dataset, legit, fraud, results = _cached_pipeline()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Raw synthetic rows", len(credit_card_data))
c2.metric("Legit (class 0)", len(legit))
c3.metric("Fraud (class 1)", len(fraud))
c4.metric("new_dataset rows", len(new_dataset))

st.subheader("Notebook-style metrics")
m1, m2 = st.columns(2)
m1.metric("Accuracy on training data", f"{results['train_acc']:.6f}")
m2.metric("Accuracy score on test data", f"{results['test_acc']:.6f}")

st.subheader("Classification report (test)")
st.code(results["report"], language="text")

st.subheader("Confusion matrix (test)")
st.dataframe(
    pd.DataFrame(
        results["cm"],
        index=["Actual 0", "Actual 1"],
        columns=["Pred 0", "Pred 1"],
    ),
    use_container_width=True,
)

st.subheader("Try one row (median feature values from training)")
feat_names = results["feature_names"]
row = {}
cols = st.columns(min(6, len(feat_names)))
for i, name in enumerate(feat_names):
    col = cols[i % len(cols)]
    default = float(results["X_train"][name].median())
    row[name] = col.number_input(name, value=default, format="%.6f", key=f"in_{name}")

if st.button("Predict"):
    X_one = pd.DataFrame([row])[feat_names]
    pred = int(results["model"].predict(X_one)[0])
    proba = results["model"].predict_proba(X_one)[0]
    st.success(
        f"**Prediction:** {'Fraud' if pred == 1 else 'Legit'} (class {pred}) — "
        f"P(0)={proba[0]:.4f}, P(1)={proba[1]:.4f}"
    )

st.divider()
st.markdown(
    "**If the link fails:** install deps and start the server on your machine "
    "(the app must be running for `localhost` to work).\n\n"
    "```bash\npython3 -m pip install streamlit pandas numpy scikit-learn\n"
    "streamlit run collllab1_app.py --server.port 8503\n```\n\n"
    "Open **http://localhost:8503** in Chrome/Safari/Edge."
)

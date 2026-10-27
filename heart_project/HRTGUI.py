"""
Heart disease prediction - local web app (Streamlit).

Run:
    pip install streamlit pandas numpy scikit-learn
    streamlit run app.py

Keep processed.cleveland.data in the SAME folder as this file.
Research/learning project only. NOT medical advice.
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_FILE = Path(__file__).parent / "processed.cleveland.data"
COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal", "target"]
FEATURES = COLS[:-1]


MODEL_FILE = Path(__file__).parent / "heart_models.joblib"


@st.cache_resource
def load_and_train():
    """Use the models saved by the notebook if heart_models.joblib exists,
    otherwise train fresh from the data file. Returns (..., source_text)."""
    if MODEL_FILE.exists():
        try:
            saved = joblib.load(MODEL_FILE)
            return (saved["models"], saved["X_test"], saved["y_test"], saved["accuracy"],
                    "Loaded trained models from heart_models.joblib (saved by the notebook)")
        except Exception as e:
            note = (f"heart_models.joblib could not be loaded ({type(e).__name__}). This usually means the "
                    "notebook and the app use different Python or library versions, so the models were "
                    "trained fresh from the data file instead.")
            return (*train_from_data(), note)
    return (*train_from_data(), "Trained fresh from processed.cleveland.data (no heart_models.joblib found)")


def make_models():
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Random Forest": make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=200, random_state=42)),
    }


@st.cache_data
def cv_results():
    """5-fold cross-validation on the full cleaned dataset."""
    df = pd.read_csv(DATA_FILE, header=None, names=COLS)
    df = df.replace("?", np.nan).dropna().apply(pd.to_numeric)
    y = (df["target"] > 0).astype(int)
    X = df[FEATURES]
    return {name: cross_val_score(m, X, y, cv=5, scoring="accuracy") for name, m in make_models().items()}


def metrics_table(models, X_test, y_test):
    rows = {}
    for name, m in models.items():
        pred = m.predict(X_test)
        proba = m.predict_proba(X_test)[:, 1]
        rows[name] = {
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
            "ROC AUC": roc_auc_score(y_test, proba),
        }
    return pd.DataFrame(rows).T


def fig_metric_bars(table):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    table.T.plot(kind="bar", ax=ax, rot=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Test-set metrics")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def fig_confusion(models, X_test, y_test):
    fig, axes = plt.subplots(1, len(models), figsize=(7, 3.2))
    for ax, (name, m) in zip(np.atleast_1d(axes), models.items()):
        ConfusionMatrixDisplay.from_estimator(m, X_test, y_test, display_labels=["No disease", "Disease"],
                                              cmap="Blues", ax=ax, colorbar=False)
        ax.set_title(name)
    fig.tight_layout()
    return fig


def fig_roc(models, X_test, y_test):
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for name, m in models.items():
        proba = m.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_test, proba):.2f})")
    ax.plot([0, 1], [0, 1], "k--", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def fig_cv(cv):
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.boxplot(list(cv.values()))
    ax.set_xticks(range(1, len(cv) + 1))
    ax.set_xticklabels(list(cv.keys()))
    ax.set_ylabel("Accuracy")
    ax.set_title("5-fold cross-validation")
    fig.tight_layout()
    return fig


def fig_importance(models):
    rf = models["Random Forest"].named_steps["randomforestclassifier"]
    imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(5, 4))
    imp.plot(kind="barh", ax=ax)
    ax.set_title("Random Forest: which features matter most")
    fig.tight_layout()
    return fig


def train_from_data():
    df = pd.read_csv(DATA_FILE, header=None, names=COLS)
    df = df.replace("?", np.nan).dropna().apply(pd.to_numeric)
    df["target"] = (df["target"] > 0).astype(int)  # 0 = no disease, 1 = disease

    X, y = df[FEATURES], df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    models = make_models()
    for m in models.values():
        m.fit(X_train, y_train)
    accuracy = {name: m.score(X_test, y_test) for name, m in models.items()}
    return models, X_test, y_test, accuracy


def predict_table(models, X):
    """Return a table with each model's label and disease probability."""
    out = pd.DataFrame(index=X.index)
    for name, m in models.items():
        out[f"{name} - disease?"] = np.where(m.predict(X) == 1, "YES", "no")
        out[f"{name} - prob"] = (m.predict_proba(X)[:, 1] * 100).round(1)
    return out


def main():
    st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")
    st.title("Heart disease prediction: model comparison")
    st.warning("Learning/research project on a small dataset (Cleveland, ~297 patients). "
               "**Not medical advice** and not a diagnostic tool.")

    if not DATA_FILE.exists():
        st.error(f"Data file not found: {DATA_FILE.name}. Put it in the same folder as app.py.")
        st.stop()

    with st.spinner("Loading models..."):
        models, X_test, y_test, accuracy, source = load_and_train()
    st.caption(source)

    c1, c2 = st.columns(2)
    for col, (name, acc) in zip((c1, c2), accuracy.items()):
        col.metric(f"{name} test accuracy", f"{acc:.1%}")

    tab0, tab1, tab2, tab3 = st.tabs(["Model comparison", "Random test patients", "Enter a patient", "Upload a CSV"])

    # ---- Tab 0: side-by-side model comparison ----
    with tab0:
        table = metrics_table(models, X_test, y_test)
        best = table["Accuracy"].idxmax()
        st.write(f"On the 20% test set, **{best}** has the higher accuracy "
                 f"({table.loc[best, 'Accuracy']:.1%}). A single small test set is noisy, "
                 "so also look at the cross-validation result below.")
        st.dataframe(table.style.format("{:.3f}"), use_container_width=True)

        left, right = st.columns(2)
        with left:
            fig = fig_metric_bars(table); st.pyplot(fig); plt.close(fig)
        with right:
            fig = fig_roc(models, X_test, y_test); st.pyplot(fig); plt.close(fig)

        st.subheader("Confusion matrices (test set)")
        fig = fig_confusion(models, X_test, y_test); st.pyplot(fig); plt.close(fig)
        st.caption("Rows = real answer, columns = prediction. A false negative (real disease predicted "
                   "as no disease) is the more serious mistake in a medical setting.")

        st.subheader("5-fold cross-validation (more reliable)")
        with st.spinner("Running cross-validation..."):
            cv = cv_results()
        l2, r2 = st.columns(2)
        with l2:
            for name, scores in cv.items():
                st.write(f"**{name}:** mean {scores.mean():.3f}, std {scores.std():.3f}")
            gap = abs(cv["Logistic Regression"].mean() - cv["Random Forest"].mean())
            spread = max(v.std() for v in cv.values())
            st.info("The gap between the models is smaller than the fold-to-fold spread, so there is "
                    "no clear winner." if gap < spread else "The gap is larger than the fold-to-fold spread.")
            fig = fig_cv(cv); st.pyplot(fig); plt.close(fig)
        with r2:
            fig = fig_importance(models); st.pyplot(fig); plt.close(fig)

    # ---- Tab 1: random patients from the held-out test set ----
    with tab1:
        st.write("Picks random patients the models have **never seen** (the 20% test set) "
                 "and compares predictions with the real answer.")
        n = st.slider("How many patients?", 5, 30, 10)
        if st.button("Pick random patients", type="primary"):
            sample = X_test.sample(n)
            table = predict_table(models, sample)
            table.insert(0, "actual", np.where(y_test.loc[sample.index] == 1, "YES", "no"))
            for name in models:
                table[f"{name} - correct"] = table["actual"] == table[f"{name} - disease?"]
            st.dataframe(table, use_container_width=True)
            for name in models:
                ok = int(table[f"{name} - correct"].sum())
                st.write(f"**{name}:** {ok} of {n} correct in this sample")
            with st.expander("Show the patient measurements"):
                st.dataframe(sample, use_container_width=True)

    # ---- Tab 2: type in one patient ----
    with tab2:
        st.write("Enter values for one patient and see what each model predicts.")
        a, b, c = st.columns(3)
        v = {}
        v["age"] = a.number_input("Age", 20, 90, 55)
        v["sex"] = a.selectbox("Sex (1 = male, 0 = female)", [1, 0])
        v["cp"] = a.selectbox("Chest pain type (1-4)", [1, 2, 3, 4], index=3)
        v["trestbps"] = a.number_input("Resting blood pressure", 80, 220, 130)
        v["chol"] = b.number_input("Cholesterol", 100, 600, 245)
        v["fbs"] = b.selectbox("Fasting blood sugar > 120 (1 = yes)", [0, 1])
        v["restecg"] = b.selectbox("Resting ECG (0-2)", [0, 1, 2])
        v["thalach"] = b.number_input("Max heart rate", 60, 220, 150)
        v["exang"] = c.selectbox("Exercise-induced angina (1 = yes)", [0, 1])
        v["oldpeak"] = c.number_input("ST depression (oldpeak)", 0.0, 7.0, 1.0, step=0.1)
        v["slope"] = c.selectbox("ST slope (1-3)", [1, 2, 3], index=1)
        v["ca"] = c.selectbox("Major vessels colored (0-3)", [0, 1, 2, 3])
        v["thal"] = c.selectbox("Thal (3 = normal, 6 = fixed defect, 7 = reversible)", [3, 6, 7])
        if st.button("Predict", type="primary", key="predict_one"):
            row = pd.DataFrame([v])[FEATURES]
            result = predict_table(models, row)
            for name in models:
                label = result[f"{name} - disease?"].iloc[0]
                prob = result[f"{name} - prob"].iloc[0]
                msg = f"**{name}:** {'disease likely' if label == 'YES' else 'no disease predicted'} ({prob}% probability)"
                (st.error if label == "YES" else st.success)(msg)

    # ---- Tab 3: CSV of many subjects ----
    with tab3:
        st.write("Upload a CSV with these 13 columns (header row required): "
                 + ", ".join(FEATURES) + ". An optional `target` column (0/1) lets the app score the models.")
        up = st.file_uploader("CSV file", type="csv")
        if up is not None:
            data = pd.read_csv(up)
            missing = [f for f in FEATURES if f not in data.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                feats = data[FEATURES].apply(pd.to_numeric, errors="coerce")
                bad = feats.isna().any(axis=1)
                if bad.any():
                    st.warning(f"Skipped {int(bad.sum())} row(s) with missing or non-numeric values.")
                feats = feats[~bad]
                table = predict_table(models, feats)
                if "target" in data.columns:
                    actual = (pd.to_numeric(data.loc[feats.index, "target"]) > 0).astype(int)
                    table.insert(0, "actual", np.where(actual == 1, "YES", "no"))
                    for name in models:
                        acc = (table[f"{name} - disease?"] == table["actual"]).mean()
                        st.write(f"**{name}** accuracy on your file: {acc:.1%}")
                st.dataframe(table, use_container_width=True)


if __name__ == "__main__":
    main()

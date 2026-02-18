#Author: Siddi Ahmed Brahim
#GitHub link: https://github.com/Sidiahmedde
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib

from url_features import extract_url_features


CSV_PATH = "SpamAssasin.csv"  # change if needed
URL_REGEX = r'https?://[^\s]+'


def extract_first_url(text):
    if not isinstance(text, str):
        return ""
    match = re.search(URL_REGEX, text)
    return match.group(0) if match else ""


def main():
    df = pd.read_csv(CSV_PATH)

    # Extract first URL from body
    df["extracted_url"] = df["body"].apply(extract_first_url)

    # Keep only rows with URLs
    df = df[df["extracted_url"] != ""].copy()

    print("Rows with URLs:", len(df))

    # Build feature matrix
    X = pd.DataFrame([extract_url_features(u) for u in df["extracted_url"]])
    y = df["label"].astype(int)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, "url_model.joblib")
    print("Model saved as url_model.joblib")

    # Evaluate
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    print("ROC-AUC:", roc_auc_score(y_test, proba))
    print(classification_report(y_test, preds, digits=4))

    # Feature importance
    coef = pd.Series(model.coef_[0], index=X.columns).sort_values()
    print("\nTop phishing indicators:")
    print(coef.tail(8))


if __name__ == "__main__":
    main()
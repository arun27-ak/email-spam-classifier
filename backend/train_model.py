
import io
import zipfile
import urllib.request
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Dataset download URL
DATA_URL = (
    "https://archive.ics.uci.edu/"
    "ml/machine-learning-databases/00228/"
    "smsspamcollection.zip"
)

MODEL_PATH = Path("spam_model.joblib")


def load_dataset():
    """Download and load the labeled spam dataset."""

    print("Downloading dataset...")

    request = urllib.request.Request(
        DATA_URL,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        zip_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        # Find the dataset file inside the ZIP.
        filename = next(
            name for name in archive.namelist()
            if name.endswith("SMSSpamCollection")
        )

        with archive.open(filename) as file:
            data = pd.read_csv(
                file,
                sep="\t",
                header=None,
                names=["label", "message"],
                encoding="utf-8"
            )

    # Remove invalid or empty records.
    data = data.dropna(subset=["label", "message"])
    data["label"] = data["label"].str.strip().str.lower()
    data["message"] = data["message"].astype(str).str.strip()

    data = data[
        data["label"].isin(["spam", "ham"])
        & data["message"].ne("")
    ].copy()

    # Remove duplicate messages to reduce leakage
    # between training and testing.
    data = data.drop_duplicates(
        subset=["label", "message"]
    )

    if data.empty:
        raise ValueError("The dataset is empty.")

    print("\nDataset loaded successfully.")
    print("Number of messages:", len(data))
    print("\nClass counts:")
    print(data["label"].value_counts())

    return data


def train_model():
    # 2. Load data
    data = load_dataset()

    X = data["message"]
    y = data["label"]

    # 3. Split into training and testing data.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # 4. Create a text classification pipeline.
    # TF-IDF converts text into numerical features.
    # Naive Bayes learns to classify spam and ham.
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            MultinomialNB(alpha=0.5)
        )
    ])

    # 5. Train the ML model.
    print("\nTraining the model...")
    model.fit(X_train, y_train)

    # 6. Evaluate on unseen test messages.
    predictions = model.predict(X_test)

    print("\n========== MODEL RESULTS ==========")
    print(
        "Accuracy:",
        f"{accuracy_score(y_test, predictions) * 100:.2f}%"
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=["ham", "spam"],
            zero_division=0
        )
    )

    print("Confusion Matrix (rows=true, columns=predicted)")
    print("Order: ham, spam")
    print(confusion_matrix(
        y_test,
        predictions,
        labels=["ham", "spam"]
    ))

    # 7. Save the complete pipeline.
    joblib.dump(model, MODEL_PATH)

    print("\nModel saved to:", MODEL_PATH.resolve())
    print("Training completed successfully!")


if __name__ == "__main__":
    try:
        train_model()
    except Exception as error:
        print("\nERROR:", error)
        print(
            "Check your internet connection, dataset URL, "
            "and installed libraries."
        )
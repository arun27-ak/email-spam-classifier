import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path
import joblib
from flask import Flask, render_template, request
from pathlib import Path

import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_PATH = Path(__file__).parent / "spam_model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model not found. Run train_model.py first."
    )

# Load the trained pipeline once when the app starts.
model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None
    message = ""
    error = None

    if request.method == "POST":
        message = request.form.get("email", "").strip()

        if not message:
            error = "Please enter an email message."
        elif len(message) > 20000:
            error = "Please enter a message under 20,000 characters."
        else:
            # Predict spam or ham.
            prediction = model.predict([message])[0]

            # Calculate the model's estimated class probability.
            probabilities = model.predict_proba([message])[0]
            class_probabilities = dict(
                zip(model.classes_, probabilities)
            )

            result = prediction.upper()
            confidence = round(
                float(class_probabilities[prediction]) * 100, 2
            )

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        message=message,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
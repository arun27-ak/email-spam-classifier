
import joblib
from pathlib import Path

MODEL_PATH = Path("spam_model.joblib")


def predict_message(message, model):
    """Predict whether a message is spam or ham."""

    message = message.strip()

    if not message:
        print("Please enter a message.")
        return

    prediction = model.predict([message])[0]
    probabilities = model.predict_proba([message])[0]

    # Get probability corresponding to each class.
    class_probabilities = dict(
        zip(model.classes_, probabilities)
    )

    confidence = class_probabilities[prediction] * 100

    print("\n========== PREDICTION ==========")

    if prediction == "spam":
        print("Result: SPAM")
    else:
        print("Result: HAM (Legitimate message)")

    print(f"Model confidence: {confidence:.2f}%")
    print("================================")


def main():
    if not MODEL_PATH.exists():
        print("Trained model not found.")
        print("First run: python train_model.py")
        return

    # Load the trained pipeline.
    model = joblib.load(MODEL_PATH)

    print("Email Spam Classifier")
    print("Enter a message to classify.")
    print("Type 'exit' to stop.")

    while True:
        message = input("\nEnter email/message: ")

        if message.strip().lower() == "exit":
            print("Program closed.")
            break

        predict_message(message, model)


if __name__ == "__main__":
    main()
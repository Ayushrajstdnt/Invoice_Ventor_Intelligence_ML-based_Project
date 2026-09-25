import joblib
import pandas as pd
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "freight_model.pkl")


def load_model(model_path: str = MODEL_PATH):
    """Load trained model from disk."""
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model


def predict_freight_cost(input_data):
    """
    Predict freight cost for new vendor invoices using the trained model.
    """
    model = load_model()
    input_df = pd.DataFrame(input_data)

    # Feature engineering
    if "Quantity" not in input_df.columns:
        input_df["Quantity"] = 1

    input_df["Cost_Per_Unit"] = (
    input_df["Dollars"] /
    input_df["Quantity"].replace(0, 1)
    )

    input_df = input_df[["Quantity", "Dollars", "Cost_Per_Unit"]]

    # Prediction
    predictions = model.predict(input_df).round()
    input_df["Predicted_Freight"] = predictions

    return input_df


if __name__ == "__main__":
    sample_data = {
        "Quantity": [1200, 800, 500, 100],
        "Dollars": [18500, 9000, 3000, 200]
    }
    result = predict_freight_cost(sample_data)
    print(result)
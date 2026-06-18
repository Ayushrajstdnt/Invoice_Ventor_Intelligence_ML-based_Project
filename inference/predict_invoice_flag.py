import joblib
import pandas as pd
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "invoice_model.pkl"
)
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")

FEATURES = [
    "amount_difference_pct",
    "freight_pct",
    "quantity_difference_pct",
    "days_po_to_invoice",
    "total_brands"
    ]
RISK_THRESHOLD = 0.40


def load_model(model_path: str = MODEL_PATH):
    """
    Load trained invoice flag prediction model.
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)
        
    return model


def load_scaler(scaler_path: str = SCALER_PATH):
    """
    Load fitted feature scaler.
    """
    with open(scaler_path, "rb") as f:
        scaler = joblib.load(f)
        
    return scaler


def get_risk_level(prob):
    """
    Convert risk probability into a business-friendly label.
    """
    if prob < 30:
        return "🟢 Safe"

    elif prob < 70:
        return "🟡 Review"

    else:
        return "🔴 Manual Approval"
    

def predict_invoice_flag(input_data):
    """
    Predict whether an invoice should be flagged
    for manual approval.
    """
    model = load_model()
    scaler = load_scaler()

    input_df = pd.DataFrame(input_data)

    input_df["amount_difference_pct"] = abs(
        input_df["invoice_dollars"] - input_df["total_item_dollars"]
    ) / input_df["total_item_dollars"].replace(0, 1)

    input_df["quantity_difference_pct"] = abs(
        input_df["invoice_quantity"] - input_df["total_item_quantity"]
    ) / input_df["total_item_quantity"].replace(0, 1)

    input_df["freight_pct"] = (
        input_df["Freight"] / input_df["invoice_dollars"].replace(0, 1)
    )

    input_df = input_df[FEATURES]

    # scale
    input_scaled = scaler.transform(input_df)

    # predict
    probabilities = model.predict_proba(input_scaled)[:, 1]
    predictions = (probabilities >= RISK_THRESHOLD).astype(int)

    original_df = pd.DataFrame(input_data)

    result_df = original_df.copy()
    result_df["Predicted_Flag"] = predictions
    result_df["Risk_Probability"] = (probabilities * 100).round(2)

    result_df["Risk_Level"] = (
    result_df["Risk_Probability"]
    .apply(get_risk_level)
    )

    return result_df


if __name__ == "__main__":
    sample_data = {
    "invoice_quantity": [100],
    "invoice_dollars": [18500],
    "Freight": [90],
    "total_brands": [12],
    "total_item_quantity": [100],
    "total_item_dollars": [17000],
    "days_po_to_invoice": [25]
    }

    prediction = predict_invoice_flag(sample_data)
    print(prediction)
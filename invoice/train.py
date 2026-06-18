import joblib
import os
import pandas as pd
import json

from model_evaluation import train_random_forest, evaluate_classifier
from data_preprocessing import (
    load_invoice_data,
    validate_data,
    clean_data,
    apply_labels,
    split_data,
    scale_features
)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = [
    "amount_difference_pct",
    "freight_pct",
    "quantity_difference_pct",
    "days_po_to_invoice",
    "total_brands"
]

TARGET = "flag_invoice"

def main():
    """
    Train, evaluate, and save the invoice manual approval
    prediction model and related artifacts.
    """
    # load data
    df = load_invoice_data()

    validate_data(df)

    df = clean_data(df)

    df = apply_labels(df)

    print("\nTarget Distribution:")
    print(df[TARGET].value_counts())

    # prepare data
    X_train, X_test, y_train, y_test = split_data(df, FEATURES, TARGET)

    # Fit scaler properly
    X_train_scaled, X_test_scaled = scale_features(
        X_train,
        X_test,
        os.path.join(MODEL_DIR, "scaler.pkl")
    )
    
    # train model
    grid_search = train_random_forest(X_train_scaled, y_train)

    best_model = grid_search.best_estimator_

    feature_importance_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": best_model.feature_importances_
    })

    feature_importance_df = (
        feature_importance_df
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    feature_importance_df.to_csv(
        os.path.join(MODEL_DIR, "invoice_feature_importance.csv"),
        index=False
    )

    print("\nFeature Importance:")
    print(feature_importance_df)

    metrics = evaluate_classifier(
    best_model,
    X_test_scaled,
    y_test,
    "Random Forest Classifier"
    )

    with open(
        os.path.join(MODEL_DIR, "invoice_metrics.json"),
        "w"
    ) as f:
        json.dump(metrics, f, indent=4)

    # save model
    joblib.dump(best_model, os.path.join(MODEL_DIR, "invoice_model.pkl"))
    print(f"\nModel saved successfully: "f"{os.path.join(MODEL_DIR, 'invoice_model.pkl')}")

if __name__ == "__main__":
    main()

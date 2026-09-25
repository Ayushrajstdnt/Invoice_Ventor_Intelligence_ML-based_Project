"""
Bootstrap script to generate model .pkl files for deployment.

Run this once locally to create the models that will be committed to git.
Since the original training database (data/inventory.db) is not tracked in git,
this script creates models trained on representative synthetic data that match
the original model's characteristics (Linear Regression for freight, 
Random Forest Classifier for invoice risk).

Usage:
    python generate_models.py
"""

import joblib
import numpy as np
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

np.random.seed(42)


def generate_freight_model():
    """
    Generate and save the freight cost prediction model.
    Best model from original training was Linear Regression with:
      - MAE: 24.40, RMSE: 124.40, R²: 97.0%
      - Features: Quantity, Dollars, Cost_Per_Unit
      - Freight ≈ 0.5% of Dollars (dominant feature, importance 99.67%)
    """
    n_samples = 5000

    quantity = np.random.randint(1, 5000, n_samples).astype(float)
    dollars = np.random.uniform(50, 50000, n_samples)
    cost_per_unit = dollars / np.maximum(quantity, 1)

    # Freight is primarily driven by dollars (~0.5% of invoice value)
    freight = (
        0.005 * dollars
        + 0.001 * quantity
        + np.random.normal(0, 15, n_samples)
    )
    freight = np.maximum(freight, 0)

    X = pd.DataFrame({
        "Quantity": quantity,
        "Dollars": dollars,
        "Cost_Per_Unit": cost_per_unit
    })

    model = LinearRegression()
    model.fit(X, freight)

    model_path = os.path.join(MODEL_DIR, "freight_model.pkl")
    joblib.dump(model, model_path)
    print(f"Freight model saved: {model_path}")

    # Quick validation
    sample = pd.DataFrame({
        "Quantity": [1200.0],
        "Dollars": [18500.0],
        "Cost_Per_Unit": [18500.0 / 1200.0]
    })
    pred = model.predict(sample)
    print(f"  Sample prediction (Qty=1200, $18500): ${pred[0]:.2f}")


def generate_invoice_model():
    """
    Generate and save the invoice risk classification model + scaler.
    Best model from original training was Random Forest Classifier with:
      - Accuracy: 96.57%, Precision: 99.62%, Recall: 87.54%, F1: 93.19%
      - Features: amount_difference_pct, freight_pct,
                  quantity_difference_pct, days_po_to_invoice, total_brands
    """
    n_samples = 5000

    amount_difference_pct = np.random.exponential(0.08, n_samples)
    freight_pct = np.random.exponential(0.05, n_samples)
    quantity_difference_pct = np.random.exponential(0.10, n_samples)
    days_po_to_invoice = np.random.uniform(0, 60, n_samples)
    total_brands = np.random.randint(1, 30, n_samples).astype(float)

    # Flag based on original business rules
    flag = (
        (amount_difference_pct > 0.10)
        | (quantity_difference_pct > 0.25)
        | (days_po_to_invoice > 45)
        | (freight_pct > 0.15)
    ).astype(int)

    X = pd.DataFrame({
        "amount_difference_pct": amount_difference_pct,
        "freight_pct": freight_pct,
        "quantity_difference_pct": quantity_difference_pct,
        "days_po_to_invoice": days_po_to_invoice,
        "total_brands": total_brands
    })

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved: {scaler_path}")

    # Train classifier
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled, flag)

    model_path = os.path.join(MODEL_DIR, "invoice_model.pkl")
    joblib.dump(model, model_path)
    print(f"Invoice model saved: {model_path}")

    # Quick validation
    sample = pd.DataFrame({
        "amount_difference_pct": [0.088],
        "freight_pct": [0.0049],
        "quantity_difference_pct": [0.0],
        "days_po_to_invoice": [25.0],
        "total_brands": [12.0]
    })
    sample_scaled = scaler.transform(sample)
    prob = model.predict_proba(sample_scaled)[:, 1]
    print(f"  Sample risk probability: {prob[0]*100:.2f}%")


if __name__ == "__main__":
    print("Generating models for deployment...\n")
    generate_freight_model()
    print()
    generate_invoice_model()
    print("\nAll models generated successfully!")
    print("Now commit and push the models/ directory to deploy.")

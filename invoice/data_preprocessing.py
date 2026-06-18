import sqlite3
import pandas as pd
from pathlib import Path
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "inventory.db"

def load_invoice_data():
    """
    Load and prepare invoice-related data from the SQLite database.
    """
    conn = sqlite3.connect(DB_PATH)

    query = """
    WITH purchase_agg AS (
                SELECT
                    p.PONumber,
                    COUNT(DISTINCT p.Brand) AS total_brands,
                    SUM(p.Quantity) AS total_item_quantity,
                    SUM(p.Dollars) AS total_item_dollars,
                    AVG(julianday(p.ReceivingDate) - julianday(p.PODate)) AS avg_receiving_delay
                FROM purchases p
                GROUP BY p.PONumber
    )
                    
                SELECT
                    vi.PONumber,
                    vi.Quantity AS invoice_quantity,
                    vi.Dollars AS invoice_dollars,
                    vi.Freight,
                    (julianday(vi.InvoiceDate) - julianday(vi.PODate)) AS days_po_to_invoice,
                    (julianday(vi.PayDate) - julianday(vi.InvoiceDate)) AS days_to_pay,
                    pa.total_brands,
                    pa.total_item_quantity,
                    pa.total_item_dollars,
                    pa.avg_receiving_delay

                FROM vendor_invoice vi
                LEFT JOIN purchase_agg pa
                    ON vi.PONumber = pa.PONumber

    """

    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def validate_data(df):
    """
    Check dataset quality and return validation statistics.
    """
    print("\nDataset Validation")
    print("-" * 50)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print(f"\nDuplicate Rows: {df.duplicated().sum()}")

    print("\nNegative Value Counts:")

    numeric_cols = [
        "invoice_quantity",
        "invoice_dollars",
        "Freight",
        "days_po_to_invoice",
        "days_to_pay"
    ]

    for col in numeric_cols:
        print(f"{col}: {(df[col] < 0).sum()}")


def clean_data(df):
    """
    Remove duplicate and invalid invoice records.
    """
    df = df.copy()
    df = df.drop_duplicates()
    df = df[
        (df["invoice_quantity"] > 0)
        &
        (df["invoice_dollars"] > 0)
        &
        (df["Freight"] >= 0)
    ]

    return df


def apply_labels(df):
    """
    Create manual approval flag labels using business rules.
    """
    df["amount_difference_pct"] = (
        abs(df["invoice_dollars"] - df["total_item_dollars"])
        / df["total_item_dollars"].replace(0, 1)
    )

    df["freight_pct"] = (
        df["Freight"] / df["invoice_dollars"].replace(0, 1)
    )

    df["quantity_difference_pct"] = (
        abs(df["invoice_quantity"] - df["total_item_quantity"])
        / df["total_item_quantity"].replace(0, 1)
    )

    df["flag_invoice"] = (
        (df["amount_difference_pct"] > 0.10) |
        (df["avg_receiving_delay"] > 15) |
        (df["days_to_pay"] > 45) |
        (df["freight_pct"] > 0.15)
    ).astype(int)

    return df


def split_data(df, features, target):
    """
    Split invoice data into training and testing sets.
    """
    X = df[features]
    y = df[target]

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def scale_features(X_train, X_test, scaler_path):
    """
    Scale features and save the fitted scaler for inference.
    """
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, scaler_path)

    return X_train_scaled, X_test_scaled


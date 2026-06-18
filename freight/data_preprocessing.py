import sqlite3
from sklearn.model_selection import train_test_split
import pandas as pd


# Data loading
def load_vendor_invoice_data(db_path: str):
    """
    Load vendor invoice data from SQLite database.
    """
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM vendor_invoice"
    df = pd.read_sql_query(query, conn)
    conn.close()

    return df


# Feature engineering
def prepare_features(df):
    """
    Create features and target variable for freight prediction.
    """
    df = df.copy()
    df["Cost_Per_Unit"] = df["Dollars"] / df["Quantity"].replace(0, 1)
    X = df[["Quantity", "Dollars", "Cost_Per_Unit"]]
    y = df["Freight"]

    return X, y


# Data validation
def validate_data(df):
    """
    Check dataset quality and return validation statistics.
    """
    missing = df.isnull().sum()
    duplicate_count = df.duplicated().sum()
    negative_counts = {
        col: (df[col] < 0).sum()
        for col in ["Quantity", "Dollars", "Freight"]
    }

    return {
        "missing_values": missing.to_dict(),
        "duplicate_rows": duplicate_count,
        "negative_values": negative_counts
    }


# Data cleaning
def clean_data(df):
    """
    Remove duplicate and invalid invoice records.
    """
    df = df.copy()
    df = df.drop_duplicates()
    df = df[
        (df["Quantity"] > 0)
        & (df["Dollars"] > 0)
        & (df["Freight"] >= 0)
    ]
    
    return df


# Data splitting
def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    """

    return train_test_split(X, y, test_size=test_size, random_state=random_state)
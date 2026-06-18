import json
import joblib

from pathlib import Path
from data_preprocessing import (
    load_vendor_invoice_data,
    prepare_features,
    split_data,
    validate_data,
    clean_data
)
from model_evaluation import (
    train_linear_regression,
    train_decision_tree,
    train_random_forest,
    train_tuned_random_forest,
    evaluate_model,
    cross_validate_model,
    get_feature_importance
)


def main():
    """
    Train, evaluate, and save the best freight cost prediction model.
    """

    base_dir = Path(__file__).resolve().parent.parent

    db_path = base_dir / "data" / "inventory.db"

    model_dir = base_dir / "models"
    model_dir.mkdir(exist_ok=True)

    # Load and validate data
    df = load_vendor_invoice_data(str(db_path))
    validate_data(df)

    # Clean data
    df = clean_data(df)

    # Feature engineering
    X, y = prepare_features(df)

    # Train-test split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train models
    lr_model = train_linear_regression(X_train, y_train)
    dt_model = train_decision_tree(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    tuned_rf_model = train_tuned_random_forest(X_train, y_train)

    # Cross-validation
    lr_cv_mae = cross_validate_model(
        lr_model, X, y, "Linear Regression"
    )

    dt_cv_mae = cross_validate_model(
        dt_model, X, y, "Decision Tree Regression"
    )

    rf_cv_mae = cross_validate_model(
        rf_model, X, y, "Random Forest Regression"
    )

    tuned_rf_cv_mae = cross_validate_model(
        tuned_rf_model, X, y, "Tuned Random Forest"
    )

    # Evaluate models
    results = []

    for model, model_name, cv_mae in [
        (lr_model, "Linear Regression", lr_cv_mae),
        (dt_model, "Decision Tree Regression", dt_cv_mae),
        (rf_model, "Random Forest Regression", rf_cv_mae),
        (tuned_rf_model, "Tuned Random Forest", tuned_rf_cv_mae)
    ]:

        result = evaluate_model(
            model,
            X_test,
            y_test,
            model_name
        )

        result["cv_mae"] = cv_mae
        results.append(result)

    # Select best model
    best_model_info = min(
        results,
        key=lambda x: x["mae"]
    )

    best_model_name = best_model_info["model_name"]

    # Feature importance
    feature_importance_df = get_feature_importance(
        tuned_rf_model,
        X.columns
    )

    feature_importance_df.to_csv(
        model_dir / "freight_feature_importance.csv",
        index=False
    )

    print("\nFeature Importance:")
    print(feature_importance_df)

    # Save metrics
    metrics_summary = {
        result["model_name"]: {
            "mae": round(result["mae"], 2),
            "rmse": round(result["rmse"], 2),
            "r2": round(result["r2"], 2),
            "cv_mae": round(result["cv_mae"], 2)
        }
        for result in results
    }

    with open(model_dir / "freight_metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=4)

    with open(model_dir / "best_freight_model.json", "w") as f:
        json.dump(best_model_info, f, indent=4)

    # Save best model
    best_model = {
        "Linear Regression": lr_model,
        "Decision Tree Regression": dt_model,
        "Random Forest Regression": rf_model,
        "Tuned Random Forest": tuned_rf_model
    }[best_model_name]

    joblib.dump(
        best_model,
        model_dir / "freight_model.pkl"
    )

    print(f"\nBest model saved: {best_model_name}")


if __name__ == "__main__":
    main()
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_linear_regression(X_train, y_train):
    """
    Train a Linear Regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model


def train_decision_tree(X_train, y_train, max_depth=5):
    """
    Train a Decision Tree Regressor.
    """
    model = DecisionTreeRegressor(
        max_depth=max_depth, random_state=42
    )
    model.fit(X_train, y_train)

    return model


def train_random_forest(X_train, y_train, max_depth=5, n_estimators=100):
    """
    Train a Random Forest Regressor.
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    return model


def train_tuned_random_forest(X_train, y_train):
    """
    Train a tuned Random Forest model using GridSearchCV.
    """
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "min_samples_split": [2, 5]
    }
    grid = GridSearchCV(
        estimator=RandomForestRegressor(
            random_state=42,
            n_jobs=-1
        ),
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    return grid.best_estimator_


def cross_validate_model(model, X, y, model_name):
    """
    Perform cross-validation and return average MAE.
    """
    scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="neg_mean_absolute_error"
    )

    mae_scores = -scores

    print(f"\n{'-'*50}")
    print(f"{model_name} - Cross Validation")
    print(f"{'-'*50}")

    print(f"CV MAE Scores: {mae_scores}")
    print(
        f"Average CV MAE: {mae_scores.mean():.2f}"
    )

    return mae_scores.mean()


def get_feature_importance(model, feature_names):
    """
    Generate feature importance rankings for tree-based models.
    """
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })
    importance_df = (
        importance_df
        .sort_values(
            by="Importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return importance_df


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Evaluate a regression model and return performance metrics.
    """
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds) * 100

    print(f"\n{model_name} Performance:")
    print(f"MAE : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R² : {r2:.2f}%")

    return {
        "model_name": model_name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }


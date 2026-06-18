from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    make_scorer
)


def train_random_forest(X_train, y_train):
    """
    Train a Random Forest classifier using GridSearchCV
    to identify invoices requiring manual review.
    """
    rf = RandomForestClassifier(
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
    )

    param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 15, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ['sqrt'],
    "criterion": ['gini', 'entropy']
    }

    # Scorer
    scorer = make_scorer(f1_score, average='binary')

    # Grid Search
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring=scorer,
        cv=5,
        verbose=2,
        n_jobs=-1
    )

    # Train
    grid_search.fit(X_train, y_train)

    return grid_search


def evaluate_classifier(model, X_test, y_test, model_name):
    """
    Evaluate classifier performance using accuracy,
    precision, recall, and F1 score.
    """
    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print(f"\n{model_name} Performance")
    print(f"Accuracy : {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall   : {recall:.2f}")
    print(f"F1 Score : {f1:.2f}")

    return {
        "model_name": model_name,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4)
    }

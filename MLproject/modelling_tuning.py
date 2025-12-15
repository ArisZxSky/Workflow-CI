import sys
import mlflow
import pandas as pd
import numpy as np
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, root_mean_squared_error


def main(dataset_path, n_iter, cv):
    data = pd.read_csv(dataset_path)

    X = data.drop("charges", axis=1)
    y = data["charges"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    input_example = X_train.iloc[:5]

    rf = RandomForestRegressor(
        random_state=42,
        n_jobs=-1
    )

    param_dist = {
        "n_estimators": np.arange(200, 1000, 200),
        "max_depth": [None, 10, 20, 30, 50],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"]
    }

    search = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="r2",
        n_jobs=-1,
        random_state=42
    )

    with mlflow.start_run():
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)

        mlflow.log_metric("r2_score", r2_score(y_test, y_pred))
        mlflow.log_metric("rmse", root_mean_squared_error(y_test, y_pred))

        mlflow.log_params(search.best_params_)
        mlflow.sklearn.log_model(
            best_model,
            artifact_path="model",
            input_example=input_example
        )


if __name__ == "__main__":
    dataset = sys.argv[1]
    n_iter = int(sys.argv[2])
    cv = int(sys.argv[3])

    main(dataset, n_iter, cv)
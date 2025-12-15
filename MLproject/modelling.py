import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import random
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import warnings

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Load dataset preprocessing
    file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "train_preprocessing.csv"
    )

    data = pd.read_csv(file_path)

    # Split data
    X = data.drop("charges", axis=1)
    y = data["charges"]

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

    input_example = X_train.head(5)

    # Model regression
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Tracking mlflow
    with mlflow.start_run():
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", 100)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )
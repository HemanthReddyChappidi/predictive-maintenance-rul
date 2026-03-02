import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import joblib
import mlflow
import mlflow.xgboost

mlflow.set_tracking_uri("file://" + str(Path("./mlruns").resolve()))
mlflow.set_experiment("Predictive-Maintenance-RUL")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "train_features.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "xgboost_rul_model.pkl"


def load_data():
    print("Loading feature dataset...")
    return pd.read_csv(DATA_FILE)


def split_data(df):
    # Drop non-feature columns
    drop_cols = ["engine_id", "cycle", "max_cycle"]
    X = df.drop(columns=drop_cols + ["RUL"])
    y = df["RUL"]

    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(X_train, y_train):
    print("Training XGBoost model...")

    params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    }

    model = XGBRegressor(**params, n_jobs=-1)
    model.fit(X_train, y_train)

    return model, params


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)

    print("\nModel Performance")
    print("-----------------")
    print(f"RMSE: {rmse:.2f} cycles")
    print(f"MAE : {mae:.2f} cycles")

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)


def save_model(model):
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    # Start MLflow run for entire pipeline
    with mlflow.start_run():

        model, params = train_model(X_train, y_train)

        # log hyperparameters
        mlflow.log_params(params)

        # evaluate and log metrics
        evaluate(model, X_test, y_test)

        # log model artifact
        mlflow.xgboost.log_model(model, "model")

        # save local copy
        save_model(model)


if __name__ == "__main__":
    main()
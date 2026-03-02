import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "train_features.csv"
MODEL_PATH = BASE_DIR / "models" / "xgboost_rul_model.pkl"
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_FILE)
    drop_cols = ["engine_id", "cycle", "max_cycle"]
    X = df.drop(columns=drop_cols + ["RUL"])
    y = df["RUL"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def main():
    print("Loading data and model...")
    X_train, X_test, y_train, y_test = load_data()
    model = joblib.load(MODEL_PATH)

    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"RMSE: {rmse:.2f}")

    # 1️⃣ Prediction vs Ground Truth
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, preds, alpha=0.3)
    plt.xlabel("True RUL")
    plt.ylabel("Predicted RUL")
    plt.title("Predicted vs True RUL")
    plt.savefig(REPORT_DIR / "pred_vs_true.png")
    plt.show()

    # 2️⃣ Error distribution
    errors = y_test - preds
    plt.figure(figsize=(6,4))
    plt.hist(errors, bins=50)
    plt.title("Prediction Error Distribution")
    plt.xlabel("Error (cycles)")
    plt.savefig(REPORT_DIR / "error_distribution.png")
    plt.show()


if __name__ == "__main__":
    main()
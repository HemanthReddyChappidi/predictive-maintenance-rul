import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "train_processed.csv"

WINDOW_SIZE = 30
DEVICE = "cpu"


# ---------- Prepare sequences ----------
def create_sequences(df, window=30):
    sensors = [col for col in df.columns if "sensor_" in col]

    sequences = []
    labels = []

    for engine_id, engine_df in df.groupby("engine_id"):
        engine_df = engine_df.sort_values("cycle")

        data = engine_df[sensors].values
        rul = engine_df["RUL"].values

        for i in range(len(engine_df) - window):
            sequences.append(data[i:i+window])
            labels.append(rul[i+window])

    return np.array(sequences), np.array(labels)


# ---------- LSTM model ----------
class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 64, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])


# ---------- Training ----------
def train_model(model, X_train, y_train):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(10):
        model.train()
        preds = model(X_train)
        loss = loss_fn(preds.squeeze(), y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1} Loss: {loss.item():.2f}")


def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        preds = model(X_test).squeeze()
        rmse = torch.sqrt(torch.mean((preds - y_test)**2))
        print(f"\nLSTM RMSE: {rmse:.2f} cycles")


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_FILE)

    print("Creating sequences...")
    X, y = create_sequences(df, WINDOW_SIZE)

    # scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    model = LSTMModel(input_size=X.shape[2])

    print("Training LSTM...")
    train_model(model, X_train, y_train)

    evaluate(model, X_test, y_test)


if __name__ == "__main__":
    main()
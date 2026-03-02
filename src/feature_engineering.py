import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA = BASE_DIR / "data" / "processed"

INPUT_FILE = PROCESSED_DATA / "train_processed.csv"
OUTPUT_FILE = PROCESSED_DATA / "train_features.csv"

WINDOW_SIZE = 30


def load_data():
    print("Loading processed dataset...")
    return pd.read_csv(INPUT_FILE)


def get_sensor_columns(df):
    return [col for col in df.columns if "sensor_" in col]


def add_rolling_features(engine_df, sensors):
    """
    Create rolling statistics for one engine
    """
    engine_df = engine_df.sort_values("cycle")

    for sensor in sensors:
        # rolling mean
        engine_df[f"{sensor}_mean"] = (
            engine_df[sensor].rolling(WINDOW_SIZE).mean()
        )

        # rolling std
        engine_df[f"{sensor}_std"] = (
            engine_df[sensor].rolling(WINDOW_SIZE).std()
        )

        # rolling min
        engine_df[f"{sensor}_min"] = (
            engine_df[sensor].rolling(WINDOW_SIZE).min()
        )

        # rolling max
        engine_df[f"{sensor}_max"] = (
            engine_df[sensor].rolling(WINDOW_SIZE).max()
        )

    return engine_df


def create_features(df):
    sensors = get_sensor_columns(df)

    print("Generating rolling features per engine...")
    engines = []

    for engine_id, engine_df in tqdm(df.groupby("engine_id")):
        engine_features = add_rolling_features(engine_df, sensors)
        engines.append(engine_features)

    df = pd.concat(engines)

    # Drop rows with NaN created by rolling window
    df = df.dropna()

    return df


def save_features(df):
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved feature dataset to {OUTPUT_FILE}")


def main():
    df = load_data()
    df_features = create_features(df)
    save_features(df_features)


if __name__ == "__main__":
    main()
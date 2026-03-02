import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA = BASE_DIR / "data" / "raw"
PROCESSED_DATA = BASE_DIR / "data" / "processed"

PROCESSED_DATA.mkdir(parents=True, exist_ok=True)


def load_raw_data():
    """Load NASA turbofan dataset"""
    
    index_names = ["engine_id", "cycle"]
    settings = ["op_setting_1", "op_setting_2", "op_setting_3"]
    sensors = [f"sensor_{i}" for i in range(1, 22)]
    columns = index_names + settings + sensors

    train_path = RAW_DATA / "train_FD001.txt"

    df = pd.read_csv(
        train_path,
        sep=r"\s+",
        header=None
    )

    df.columns = columns
    return df


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Remaining Useful Life (RUL)"""
    
    max_cycles = df.groupby("engine_id")["cycle"].max()
    df["max_cycle"] = df["engine_id"].map(max_cycles)
    df["RUL"] = df["max_cycle"] - df["cycle"]

    return df


def save_processed(df: pd.DataFrame):
    output_path = PROCESSED_DATA / "train_processed.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")


def main():
    print("Loading raw data...")
    df = load_raw_data()

    print("Adding RUL target...")
    df = add_rul(df)

    print("Saving processed dataset...")
    save_processed(df)

    print("Done! Dataset ready for feature engineering.")


if __name__ == "__main__":
    main()
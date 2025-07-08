import os
import glob
import pickle

import pandas as pd

CHECKPOINT_DIR = "src/extract/"
CHECKPOINT_PREFIX = "dataframe_checkpoint_"


def find_latest_checkpoint(directory: str = CHECKPOINT_DIR, prefix: str = CHECKPOINT_PREFIX) -> str:
    pattern = os.path.join(directory, f"{prefix}*.pickle")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError("No checkpoint files found.")
    files.sort(key=lambda x: int(os.path.splitext(x)[0].split("_")[-1]))


def load_checkpoint(checkpoint_path: str) -> pd.DataFrame:
    print(f"Loading checkpoint from {checkpoint_path}")
    with open(checkpoint_path, "rb") as f:
        df = pickle.load(f)
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Loaded object is not a pandas DataFrame.")
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns.")
    return df

import pickle
import random
import pandas as pd
from pycparser.c_ast import Node
try:
    from src.utils import load_checkpoint
except ImportError:
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import load_checkpoint

def main() -> None:
    df = load_checkpoint("src/extract/dataframe_checkpoint_20.pickle")
    n_vuln = (df["target"] == 1).sum()
    n_nonvuln = (df["target"] == 0).sum()
    print(f"Vulnerable samples: {n_vuln}")
    print(f"Non-vulnerable samples: {n_nonvuln}")
    if len(df) == 0:
        print("No samples in checkpoint.")
        return
    idx = random.randint(0, len(df) - 1)
    print("\nRandom sample:")
    print("x_string:\n", df.iloc[idx]["x_string"])
    print("\nx_ast:")
    print(df.iloc[idx]["x_ast"].show())


if __name__ == "__main__":
    main()

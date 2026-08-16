import ast
import gzip
from pathlib import Path

import pandas as pd

def load_data(file_path: str | Path) -> pd.DataFrame:
    rows = []

    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            rows.append(ast.literal_eval(line))

    return pd.DataFrame(rows)
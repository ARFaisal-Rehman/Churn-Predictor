"""Convert the UCI Online Retail II workbook (two yearly sheets) into one CSV.

    python scripts/prepare_data.py path/to/online_retail_II.xlsx

Download: https://archive.ics.uci.edu/dataset/502/online+retail+ii (CC BY 4.0).
Reading the 45 MB workbook takes a few minutes; the CSV loads in seconds after that.
"""
import sys
from pathlib import Path

import pandas as pd

src = Path(sys.argv[1])
dst = Path(__file__).resolve().parents[1] / "data" / "online_retail_ii.csv"
sheets = pd.read_excel(src, sheet_name=None, dtype=str)
df = pd.concat(sheets.values(), ignore_index=True)
dst.parent.mkdir(exist_ok=True)
df.to_csv(dst, index=False)
print(f"{len(df):,} rows from sheets {list(sheets)} -> {dst}")

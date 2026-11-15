"""Generate a synthetic customer dataset and save it as data/customers.csv.

Usage:
    python src/generate_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# (age, income in k, spending score, yearly visits, number of customers)
GROUPS = [
    (24, 30, 80, 40, 60),   # young, low income, spends a lot
    (45, 95, 85, 30, 50),   # high income, high spending
    (50, 90, 20, 8, 50),    # high income, low spending
    (30, 35, 25, 10, 50),   # low income, low spending
    (38, 60, 50, 20, 40),   # average customers
]


def make_customers(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for age, income, spend, visits, n in GROUPS:
        parts.append(pd.DataFrame({
            "Age": rng.normal(age, 4, n).clip(18, 70).round(),
            "AnnualIncome_k": rng.normal(income, 8, n).clip(10, 150).round(1),
            "SpendingScore": rng.normal(spend, 8, n).clip(1, 100).round(),
            "AnnualVisits": rng.normal(visits, 5, n).clip(1, 80).round(),
        }))
    df = pd.concat(parts, ignore_index=True)
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)
    df.insert(0, "CustomerID", range(1, len(df) + 1))
    return df


def main() -> None:
    out_dir = ROOT / "data"
    out_dir.mkdir(exist_ok=True)
    df = make_customers()
    df.to_csv(out_dir / "customers.csv", index=False)
    print(f"Saved {out_dir / 'customers.csv'} with shape {df.shape}")


if __name__ == "__main__":
    main()

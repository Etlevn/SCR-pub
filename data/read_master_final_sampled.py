from pathlib import Path

import pandas as pd


def load_master_sampled() -> pd.DataFrame:
    """
    Read data/master_final_sampled.dta and return a DataFrame.
    """
    project_dir = Path(__file__).resolve().parent
    data_path = project_dir / "master_final_sampled.dta"

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_stata(data_path)
    return df


def main() -> None:
    df = load_master_sampled()

    print("=== Overview ===")
    print(f"Rows, columns: {df.shape}")

    print("\n=== Columns ===")
    print(list(df.columns))

    print("\n=== First 5 rows ===")
    print(df.head())

    print("\n=== Details (dtypes, missing values, and related info) ===")
    df.info()


if __name__ == "__main__":
    main()


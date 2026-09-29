"""Clean the Telco churn data and write reproducible train/test CSV files."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "data" / "churn.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
TARGET = "Churn"


def prepare_datasets(input_path: Path, output_dir: Path, test_size: float, random_state: int) -> tuple[Path, Path]:
    """Preprocess the source CSV and save stratified train and test CSVs."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1 (exclusive).")

    df = pd.read_csv(input_path)
    if TARGET not in df.columns:
        raise ValueError(f"Expected target column {TARGET!r} in {input_path}.")

    # Remove the identifier and normalize fields used by the existing training code.
    df = df.drop(columns=["customerID"], errors="ignore")
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    target_values = df[TARGET].map({"Yes": 1, "No": 0})
    if target_values.isna().any():
        raise ValueError(f"{TARGET} must contain only 'Yes' or 'No' values.")
    df[TARGET] = target_values.astype("int8")

    # Encode categories once before splitting so train and test have identical columns.
    df = pd.get_dummies(df, drop_first=True, dtype="int8")
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    return train_path, test_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to source churn CSV.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for output CSVs.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction assigned to the test dataset (default: 0.2).")
    parser.add_argument("--random-state", type=int, default=42, help="Seed for a reproducible split (default: 42).")
    args = parser.parse_args()

    train_path, test_path = prepare_datasets(args.input, args.output_dir, args.test_size, args.random_state)
    print(f"Created {train_path}")
    print(f"Created {test_path}")


if __name__ == "__main__":
    main()

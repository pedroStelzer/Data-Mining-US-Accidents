import kagglehub
import polars as pl
import pandas as pd
from typing import Tuple
from .cleaning import clean_training_data

def download_dataset(dataset: str = "sobhanmoosavi/us-accidents") -> str:
    """Download dataset and return local path."""
    return kagglehub.dataset_download(dataset)

def read_polars_csv(file: str) -> pl.DataFrame:
    """Read CSV into a Polars DataFrame."""
    return pl.read_csv(file)

def sample_and_split(df: pl.DataFrame, sample_fraction: float = 0.05, seed: int = 42
                    ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Sample dataframe and produce (df_train, df_val, df_test_original)."""
    sampled = df.sample(fraction=sample_fraction, seed=seed)
    n1 = int(0.8 * sampled.height)
    df_train_original = sampled.slice(0, n1)
    df_test_original = sampled.slice(n1)

    n2 = int(0.8 * df_train_original.height)
    df_train = df_train_original.slice(0, n2)
    df_val = df_train_original.slice(n2)

    return df_train, df_val, df_train_original, df_test_original

def polars_to_pandas_xy(df: pl.DataFrame, target_col: str = "Severity") -> Tuple[pd.DataFrame, pd.Series]:
    """Convert Polars DF to pandas X (DataFrame) and y (Series)."""
    pdf = df.to_pandas()
    y = pdf[target_col].copy()
    X = pdf.drop(columns=[target_col])
    return X, y

def load_data(sample_fraction: float = 0.05, seed: int = 42, dataset: str = "sobhanmoosavi/us-accidents"):
    """High-level loader for training code.
    Returns: X_train_pd, X_test_pd, y_train_pd, y_test_pd
    """
    path = download_dataset(dataset)
    file = f"{path}/US_Accidents_March23.csv"

    df = read_polars_csv(file)
    df_train, df_val, df_train_original, df_test_original = sample_and_split(df, sample_fraction, seed)

    X_train_pd, y_train_pd = polars_to_pandas_xy(df_train)
    X_test_pd, y_test_pd = polars_to_pandas_xy(df_val)

    # Clean only training data (keeps original behaviour)
    X_train_pd, y_train_pd = clean_training_data(X_train_pd, y_train_pd)

    return X_train_pd, X_test_pd, y_train_pd, y_test_pd
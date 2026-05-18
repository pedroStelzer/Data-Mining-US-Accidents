import kagglehub
import polars as pl
import pandas as pd
import shutil
import zipfile
from pathlib import Path
from typing import Tuple, Union
from sklearn.model_selection import train_test_split
from .cleaning import clean_training_data


def _get_cache_dir(dataset: str) -> Path:
    safe_name = dataset.replace("/", "_")
    return Path.home() / ".cache" / "kagglehub" / safe_name


def _find_csv_file(path: Path) -> Path:
    if path.is_dir():
        csv_files = sorted(path.rglob("*.csv"))
        if csv_files:
            return csv_files[0]
    elif path.is_file():
        if path.suffix.lower() == ".csv":
            return path

        extract_dir = path.parent / path.stem
        extract_dir.mkdir(parents=True, exist_ok=True)

        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path, "r") as archive:
                archive.extractall(extract_dir)
        else:
            try:
                shutil.unpack_archive(str(path), str(extract_dir))
            except (shutil.ReadError, ValueError):
                pass

        csv_files = sorted(extract_dir.rglob("*.csv"))
        if csv_files:
            return csv_files[0]

    raise FileNotFoundError(f"Nenhum arquivo CSV encontrado em {path}")


def download_dataset(dataset: str = "sobhanmoosavi/us-accidents", cache_dir: str = None) -> str:
    """Download dataset and return local cache path."""
    project_root = Path.cwd()
    local_csv = project_root / "US_Accidents_March23.csv"
    local_zip = project_root / "us-accidents.zip"
    if local_csv.exists():
        return str(project_root)
    if local_zip.exists():
        return str(local_zip)

    cache_dir = Path(cache_dir) if cache_dir else _get_cache_dir(dataset)
    cache_dir.mkdir(parents=True, exist_ok=True)

    existing_csv = list(cache_dir.rglob("*.csv"))
    if existing_csv:
        return str(cache_dir)

    try:
        result = kagglehub.dataset_download(dataset, path=str(cache_dir))
    except TypeError:
        result = kagglehub.dataset_download(dataset)

    if result:
        result_path = Path(result)
        if result_path.exists():
            return str(result_path)

    return str(cache_dir)


def load_polars_csv(
    file_path,
    fraction_sev1=1.0,
    fraction_sev2=0.0194,
    fraction_sev3=0.0925,
    fraction_sev4=0.5862,
    seed: int = 42,
):
    """Carrega e balanceia os dados usando amostragem aleatória por classe."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"O arquivo de dados não foi encontrado: {file_path}")

    lazy_df = pl.scan_csv(str(file_path))

    def sample_group(severity, fraction):
        group = lazy_df.filter(pl.col("Severity") == severity).collect()
        return group if fraction >= 1.0 else group.sample(fraction=fraction, seed=seed)

    sev1 = sample_group(1, fraction_sev1)
    sev2 = sample_group(2, fraction_sev2)
    sev3 = sample_group(3, fraction_sev3)
    sev4 = sample_group(4, fraction_sev4)

    df_balanced = pl.concat([sev1, sev2, sev3, sev4]).sample(fraction=1.0, shuffle=True, seed=seed)
    return df_balanced

def sample_and_split(
    df: pd.DataFrame,
    val_size: float = 0.2,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cria split estratificado em treino/val/test."""
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["Severity"],
        random_state=seed,
    )
    val_fraction = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_fraction,
        stratify=train_val["Severity"],
        random_state=seed,
    )
    return train, val, test

def polars_to_pandas_xy(df: Union[pl.DataFrame, pd.DataFrame], target_col: str = "Severity") -> Tuple[pd.DataFrame, pd.Series]:
    """Convert Polars or pandas DF to pandas X (DataFrame) and y (Series)."""
    if isinstance(df, pl.DataFrame):
        pdf = df.to_pandas()
    else:
        pdf = df.copy()

    y = pdf[target_col].copy()
    X = pdf.drop(columns=[target_col])

    return X, y

def load_data(
    sample_fraction: float = 1.0,
    seed: int = 42,
    dataset: str = "sobhanmoosavi/us-accidents",
    val_size: float = 0.2,
    test_size: float = 0.2,
):
    """High-level loader for training code.
    Returns: X_train_pd, X_val_pd, X_test_pd, y_train_pd, y_val_pd, y_test_pd
    """
    path = download_dataset(dataset)
    csv_file = _find_csv_file(Path(path))

    df = load_polars_csv(csv_file, seed=seed)
    pdf = df.to_pandas()

    if sample_fraction < 1.0:
        pdf = pdf.sample(frac=sample_fraction, random_state=seed)

    X_raw, y_raw = polars_to_pandas_xy(pdf)
    X_clean, y_clean = clean_training_data(X_raw, y_raw)

    cleaned_pdf = X_clean.copy()
    cleaned_pdf["Severity"] = y_clean

    train_pdf, val_pdf, test_pdf = sample_and_split(
        cleaned_pdf,
        val_size=val_size,
        test_size=test_size,
        seed=seed,
    )

    X_train_pd, y_train_pd = polars_to_pandas_xy(train_pdf)
    X_val_pd, y_val_pd = polars_to_pandas_xy(val_pdf)
    X_test_pd, y_test_pd = polars_to_pandas_xy(test_pdf)

    return X_train_pd, X_val_pd, X_test_pd, y_train_pd, y_val_pd, y_test_pd
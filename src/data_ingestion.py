import os
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


def load_polars_csv(file_path: Union[str, Path], sample_fraction: float = 1.0, seed: int = 42) -> pl.DataFrame:
    """Carrega o CSV com Polars e aplica a redução opcional do tamanho do dataset original."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"O arquivo de dados não foi encontrado: {file_path}")

    # Abre o arquivo de forma "preguiçosa" (Lazy) para otimizar memória
    lazy_df = pl.scan_csv(str(file_path))

    # Coleta os dados transformando o LazyFrame em um DataFrame real
    df = lazy_df.collect()

    # Aplica a amostragem APENAS se a fração for menor que 1.0
    if sample_fraction < 1.0:
        df = df.sample(fraction=sample_fraction, seed=seed)

    return df  # Retorna o DataFrame correto (amostrado ou completo)


def sample_and_split(
    df: pd.DataFrame,
    val_size: float = 0.2,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cria split estratificado em treino/val/test a partir do subset isolado."""
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["Severity"],
        random_state=seed,
    )
    # Ajusta a proporção da validação em relação ao bloco restante de treino
    val_fraction = val_size / (1.0 - test_size)
    
    train, val = train_test_split(
        train_val,
        test_size=val_fraction,
        stratify=train_val["Severity"],
        random_state=seed,
    )
    return train, val, test


def polars_to_pandas_xy(df: Union[pl.DataFrame, pd.DataFrame], target_col: str = "Severity") -> Tuple[pd.DataFrame, pd.Series]:
    """Converte Polars ou Pandas DF para a estrutura X (DataFrame) e y (Series)."""
    if isinstance(df, pl.DataFrame):
        pdf = df.to_pandas()
    else:
        pdf = df.copy()

    y = pdf[target_col].copy()
    X = pdf.drop(columns=[target_col])

    return X, y


def download_dataset(dataset: str) -> Path:
    downloaded_path = kagglehub.dataset_download(dataset)
    if downloaded_path is None:
        raise FileNotFoundError(f"Falha ao baixar ou localizar o dataset {dataset}")
    return Path(downloaded_path)


def load_data(
    sample_fraction: float = 1.0,
    seed: int = 42,
    dataset: str = "sobhanmoosavi/us-accidents",
    val_size: float = 0.2,
    test_size: float = 0.2,
    output_dir: str = "data/processed"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Gerenciador de dados de alto nível.
    Se encontrar os arquivos na pasta local para a fração definida, carrega direto.
    Caso contrário, faz o download, divide de forma limpa e salva localmente.
    """
    # Cria uma subpasta para cada fração de amostragem para evitar conflito de dados
    local_folder = Path(output_dir) / f"sample_{sample_fraction}"
    train_path = local_folder / "train.csv"
    val_path = local_folder / "validation.csv"
    test_path = local_folder / "test.csv"

    #  PASSO EXTRA: Verifica se os 3 arquivos já existem localmente
    if train_path.exists() and val_path.exists() and test_path.exists():
        print(f"[Cache Local] Dados encontrados em '{local_folder}'. Carregando diretamente...")
        train_pdf = pd.read_csv(train_path)
        val_pdf = pd.read_csv(val_path)
        test_pdf = pd.read_csv(test_path)
        
        X_train, y_train = polars_to_pandas_xy(train_pdf)
        X_val, y_val = polars_to_pandas_xy(val_pdf)
        X_test, y_test = polars_to_pandas_xy(test_pdf)
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    print("Arquivos locais não localizados. Iniciando download e processamento original...")
    
    # 1. Download/Cache gerenciado nativamente pelo kagglehub
    path = download_dataset(dataset)
    csv_file = _find_csv_file(Path(path))

    # 2. Leitura com amostragem inicial nos dados brutos (Sem vazamento de dados!)
    print(f" -> Extraindo {sample_fraction * 100}% do dataset completo usando Polars...")
    df = load_polars_csv(csv_file, sample_fraction=sample_fraction, seed=seed)

    # 3. Conversão para Pandas e limpeza técnica (Remoção de nulos, correções estruturais)
    X_raw, y_raw = polars_to_pandas_xy(df)
    X_clean, y_clean = clean_training_data(X_raw, y_raw)

    cleaned_pdf = X_clean.copy()
    cleaned_pdf["Severity"] = y_clean

    # 4. Divisão Estratificada Intocável (Validação e Teste protegidos)
    print(f" -> Dividindo os dados (Validação: {val_size}, Teste: {test_size})...")
    train_pdf, val_pdf, test_pdf = sample_and_split(
        cleaned_pdf,
        val_size=val_size,
        test_size=test_size,
        seed=seed,
    )

    # 5. Salva os arquivos na pasta local para blindar as próximas rodadas do pipeline
    local_folder.mkdir(parents=True, exist_ok=True)
    train_pdf.to_csv(train_path, index=False)
    val_pdf.to_csv(val_path, index=False)
    test_pdf.to_csv(test_path, index=False)
    print(f"Os 3 datasets foram persistidos com sucesso em: '{local_folder}'")

    # 6. Separação final em X e y para o retorno da função
    X_train_pd, y_train_pd = polars_to_pandas_xy(train_pdf)
    X_val_pd, y_val_pd = polars_to_pandas_xy(val_pdf)
    X_test_pd, y_test_pd = polars_to_pandas_xy(test_pdf)

    return X_train_pd, X_val_pd, X_test_pd, y_train_pd, y_val_pd, y_test_pd
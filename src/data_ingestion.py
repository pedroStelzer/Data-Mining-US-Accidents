import kagglehub
import polars as pl
import pandas as pd
import numpy as np

# baixar
path = kagglehub.dataset_download("sobhanmoosavi/us-accidents")

# arquivo
file = f"{path}/US_Accidents_March23.csv"

# carregar com pollars pois o dataset tem muitos dados
df = pl.scan_csv(file)

df_sample = df.collect().sample(fraction=0.05, shuffle=True, seed=42)

# train_size é o que vai garantir a proporção 80/20 entre treino e teste
train_size = int(0.8 * df_sample.height)

df_train_original = df_sample.slice(0, train_size)
df_test_original = df_sample.slice(train_size)

print(f"Treino: {df_train_original.shape} | Teste: {df_test_original.shape}")

# separa a coluna alvo
X_train_original = df_train_original.select(pl.exclude("Severity")).to_numpy()
y_train_original = df_train_original.select("Severity").to_numpy().ravel()

X_test_original = df_test_original.select(pl.exclude("Severity")).to_numpy()
y_test_original = df_test_original.select("Severity").to_numpy().ravel()

# Pegando os nomes das colunas do DataFrame Polars original
colunas_features = [col for col in df_train_original.columns if col != "Severity"]

# Criando os DataFrames do Pandas
X_train_pd_original = pd.DataFrame(X_train_original, columns=colunas_features)
X_test_pd_original = pd.DataFrame(X_test_original, columns=colunas_features)

# Criando as Series para o alvo (y)
y_train_pd_original = pd.Series(y_train_original, name="Severity")
y_test_pd_original = pd.Series(y_test_original, name="Severity")

print("Conversão concluída do dataframe Polars para o Pandas")

# separa os dados a partir dos dados de treino
train_size = int(0.8 * df_train_original.height)

df_train = df_train_original.slice(0, train_size)
df_test = df_train_original.slice(train_size)

print(f"Treino: {df_train.shape} | Teste: {df_test.shape}")

# separa a coluna alvo
X_train = df_train.select(pl.exclude("Severity")).to_numpy()
y_train = df_train.select("Severity").to_numpy().ravel()

X_test = df_test.select(pl.exclude("Severity")).to_numpy()
y_test = df_test.select("Severity").to_numpy().ravel()

# Pegando os nomes das colunas do DataFrame Polars original
colunas_features = [col for col in df_train_original.columns if col != "Severity"]

# Criando os DataFrames do Pandas
X_train_pd = pd.DataFrame(X_train, columns=colunas_features)
X_test_pd = pd.DataFrame(X_test, columns=colunas_features)

# Criando as Series para o alvo (y)
y_train_pd = pd.Series(y_train, name="Severity")
y_test_pd = pd.Series(y_test, name="Severity")

print("Conversão concluída do dataframe Polars para o Pandas")

def drop_by_mask(X, y, mask):
    idx = X[mask].index
    return X.drop(index=idx), y.drop(index=idx)

def remove_duplicates(X, y, subset_cols):
    df = X.copy()
    df['target'] = y.values

    df = df.drop_duplicates(subset=subset_cols, keep='first')

    y_new = df['target']
    X_new = df.drop(columns=['target'])

    return X_new, y_new

def clean_temporal_noise(X, y, min_dur=1440, low_sev=2, tolerancia_horas=4):

    df = X.copy()
    df['target'] = y.values

    # garantir datetime
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], format='ISO8601')
    df['End_Time'] = pd.to_datetime(df['End_Time'], format='ISO8601')

    # duração inicial
    duration = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60
    df['duration'] = duration

    # candidatos a problema
    mask = (df['duration'] >= min_dur) & (df['target'] <= low_sev)
    subset = df.loc[mask]

    # diferença de horário (circular)
    start_min = subset['Start_Time'].dt.hour * 60 + subset['Start_Time'].dt.minute
    end_min = subset['End_Time'].dt.hour * 60 + subset['End_Time'].dt.minute

    diff = np.abs(end_min - start_min)
    diff = np.minimum(diff, 1440 - diff)

    # separar casos
    corrigir = diff <= tolerancia_horas * 60
    remover = ~corrigir

    idx_corrigir = subset.index[corrigir]
    idx_remover = subset.index[remover]

    # corrigir
    df.loc[idx_corrigir, 'End_Time'] = (
        df.loc[idx_corrigir, 'Start_Time'] +
        pd.to_timedelta(diff[corrigir], unit='m')
    )

    # remover
    df = df.drop(index=idx_remover)

    # recalcular duração FINAL (IMPORTANTE)
    df['duration'] = (
        df['End_Time'] - df['Start_Time']
    ).dt.total_seconds() / 60

    y_new = df['target']
    X_new = df.drop(columns=['target'])

    return X_new, y_new

def filter_precipitation(X, y, threshold=1.5, max_severity=3):
    mask = (X['Precipitation(in)'] > threshold) & (y <= max_severity)
    return drop_by_mask(X, y, mask)

def filter_wind_speed(X, y, threshold=60):

    good_condition = [
        'Clear', 'Fair', 'Partly Cloudy',
        'Mostly Cloudy', 'Scattered Clouds'
    ]

    mask = (
        (X['Wind_Speed(mph)'] > threshold) &
        (X['Weather_Condition'].isin(good_condition))
    )

    return drop_by_mask(X, y, mask)

def calculate_temp_limits(df):

    df = df.copy()

    df['Temperature(F)'] = pd.to_numeric(
        df['Temperature(F)'],
        errors='coerce'
    )

    grouped = df.groupby('State')['Temperature(F)']

    Q1 = grouped.quantile(0.25)
    Q3 = grouped.quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    return lower, upper

def filter_temperature_outliers(X, y):

    lower, upper = calculate_temp_limits(X)

    lower_map = X['State'].map(lower)
    upper_map = X['State'].map(upper)

    mask = (
        (X['Temperature(F)'] < lower_map) |
        (X['Temperature(F)'] > upper_map)
    )

    return drop_by_mask(X, y, mask)

def clean_training_data(X, y):

    X, y = remove_duplicates(
        X, y,
        subset_cols=['Start_Time', 'Start_Lat', 'Start_Lng']
    )

    X, y = filter_precipitation(X, y)
    X, y = filter_wind_speed(X, y)
    X, y = filter_temperature_outliers(X, y)
    X, y = clean_temporal_noise(X, y)

    return X, y

X_train_pd, y_train_pd = clean_training_data(X_train_pd, y_train_pd)
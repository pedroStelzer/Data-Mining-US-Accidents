# ...new code...
import pandas as pd
import numpy as np
from typing import Tuple

def drop_by_mask(X: pd.DataFrame, y: pd.Series, mask: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    idx = X[mask].index
    return X.drop(index=idx), y.drop(index=idx)

def remove_duplicates(X: pd.DataFrame, y: pd.Series, subset_cols) -> Tuple[pd.DataFrame, pd.Series]:
    df = X.copy()
    df['target'] = y.values
    df = df.drop_duplicates(subset=subset_cols, keep='first')
    y_new = df['target']
    X_new = df.drop(columns=['target'])
    return X_new, y_new

def clean_temporal_noise(X: pd.DataFrame, y: pd.Series, min_dur: int = 1440, low_sev: int = 2, tolerancia_horas: int = 4
                        ) -> Tuple[pd.DataFrame, pd.Series]:
    df = X.copy()
    df['target'] = y.values

    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')

    duration = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60
    df['duration'] = duration

    mask = (df['duration'] >= min_dur) & (df['target'] <= low_sev)
    subset = df.loc[mask]

    start_min = subset['Start_Time'].dt.hour * 60 + subset['Start_Time'].dt.minute
    end_min = subset['End_Time'].dt.hour * 60 + subset['End_Time'].dt.minute

    diff = np.abs(end_min - start_min)
    diff = np.minimum(diff, 1440 - diff)

    corrigir = diff <= tolerancia_horas * 60
    remover = ~corrigir

    idx_corrigir = subset.index[corrigir]
    idx_remover = subset.index[remover]

    df.loc[idx_corrigir, 'End_Time'] = (
        df.loc[idx_corrigir, 'Start_Time'] +
        pd.to_timedelta(diff[corrigir], unit='m')
    )

    df = df.drop(index=idx_remover)

    df['duration'] = (
        df['End_Time'] - df['Start_Time']
    ).dt.total_seconds() / 60

    y_new = df['target']
    X_new = df.drop(columns=['target'])
    return X_new, y_new

def filter_precipitation(X: pd.DataFrame, y: pd.Series, threshold: float = 1.5, max_severity: int = 3
                        ) -> Tuple[pd.DataFrame, pd.Series]:
    mask = (X['Precipitation(in)'] > threshold) & (y <= max_severity)
    return drop_by_mask(X, y, mask)

def filter_wind_speed(X: pd.DataFrame, y: pd.Series, threshold: float = 60.0
                     ) -> Tuple[pd.DataFrame, pd.Series]:
    good_condition = [
        'Clear', 'Fair', 'Partly Cloudy',
        'Mostly Cloudy', 'Scattered Clouds'
    ]
    mask = (
        (X['Wind_Speed(mph)'] > threshold) &
        (X['Weather_Condition'].isin(good_condition))
    )
    return drop_by_mask(X, y, mask)

def calculate_temp_limits(df: pd.DataFrame):
    df = df.copy()
    df['Temperature(F)'] = pd.to_numeric(df['Temperature(F)'], errors='coerce')
    grouped = df.groupby('State')['Temperature(F)']
    Q1 = grouped.quantile(0.25)
    Q3 = grouped.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return lower, upper

def filter_temperature_outliers(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    lower, upper = calculate_temp_limits(X)
    lower_map = X['State'].map(lower)
    upper_map = X['State'].map(upper)
    mask = (
        (X['Temperature(F)'] < lower_map) |
        (X['Temperature(F)'] > upper_map)
    )
    return drop_by_mask(X, y, mask)

def clean_training_data(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    X, y = remove_duplicates(X, y, subset_cols=['Start_Time', 'Start_Lat', 'Start_Lng'])
    X, y = filter_precipitation(X, y)
    X, y = filter_wind_speed(X, y)
    X, y = filter_temperature_outliers(X, y)
    X, y = clean_temporal_noise(X, y)
    return X, y
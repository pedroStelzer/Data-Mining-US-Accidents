import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans

class MissingValuesHandler(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self.wind_speed_median = X['Wind_Speed(mph)'].median()
        self.num_medians = {
            col: X[col].median()
            for col in ['Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)']
        }
        self.cat_modes = {
            col: X[col].mode()[0]
            for col in [
                'Weather_Condition','Wind_Direction','Sunrise_Sunset',
                'Civil_Twilight','Nautical_Twilight','Astronomical_Twilight'
            ]
        }
        return self

    def transform(self, X):
        X = X.copy()

        num_cols = [
            'Temperature(F)', 'Humidity(%)', 'Pressure(in)',
            'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)',
            'Wind_Chill(F)'
        ]

        for col in num_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')

        datetime_cols = ['Start_Time', 'End_Time', 'Weather_Timestamp']

        for col in datetime_cols:
            X[col] = pd.to_datetime(X[col], format='ISO8601')

        X['precipitation_event'] = X['Precipitation(in)'].notnull().astype(int)
        X['Precipitation(in)'] = X['Precipitation(in)'].fillna(0)

        X['Wind_Chill(F)'] = X['Wind_Chill(F)'].fillna(X['Temperature(F)'])
        X['Wind_Speed(mph)'] = X['Wind_Speed(mph)'].fillna(self.wind_speed_median)

        for col, val in self.num_medians.items():
            X[col] = X[col].fillna(val)

        for col, val in self.cat_modes.items():
            X[col] = X[col].fillna(val)

        X[['Street','City','Zipcode','Airport_Code','Timezone']] = \
            X[['Street','City','Zipcode','Airport_Code','Timezone']].fillna('Unknown')

        return X

class DropColumns(BaseEstimator, TransformerMixin):

    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.cols, errors='ignore')

class DurationFeature(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X['duration'] = (
            pd.to_datetime(X['End_Time']) -
            pd.to_datetime(X['Start_Time'])
        ).dt.total_seconds() / 60

        return X

class TimestampFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, feature):
        self.feature = feature

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        prefix = self.feature.lower()

        # hora
        hour = X[self.feature].dt.hour
        X[f"{prefix}_hour_sin"] = np.sin(2 * np.pi * hour / 24)
        X[f"{prefix}_hour_cos"] = np.cos(2 * np.pi * hour / 24)

        # dia da semana
        day_of_week = X[self.feature].dt.dayofweek
        X[f"{prefix}_day_of_week_sin"] = np.sin(2 * np.pi * day_of_week / 7)
        X[f"{prefix}_day_of_week_cos"] = np.cos(2 * np.pi * day_of_week / 7)

        # mês
        month = X[self.feature].dt.month
        X[f"{prefix}_month_sin"] = np.sin(2 * np.pi * month / 12)
        X[f"{prefix}_month_cos"] = np.cos(2 * np.pi * month / 12)

        return X

class WindDirectionFeatures(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        direcao_para_graus = {
            'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
            'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
            'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
            'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
        }

        def converter(x):
            if pd.isna(x):
                return np.nan

            x = x.upper()

            if x in ['CALM', 'VAR', 'VARIABLE']:
                return np.nan

            mapping = {
                'NORTH': 'N', 'SOUTH': 'S',
                'EAST': 'E', 'WEST': 'W'
            }
            x = mapping.get(x, x)

            return direcao_para_graus.get(x, np.nan)

        graus = X['Wind_Direction'].apply(converter)

        X['wind_missing'] = graus.isna().astype(int)
        X['wind_sin'] = np.sin(np.radians(graus))
        X['wind_cos'] = np.cos(np.radians(graus))

        X['wind_sin'] = X['wind_sin'].fillna(0)
        X['wind_cos'] = X['wind_cos'].fillna(0)

        X.drop(columns=['Wind_Direction'], inplace=True)

        return X

class WeatherAggregation(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        def weather_group(x):
            if pd.isna(x):
                return 'Unknown'

            x = x.lower()

            if 'thunder' in x or 't-storm' in x or 'storm' in x:
                return 'Storm'
            elif any(word in x for word in ['snow', 'sleet', 'ice', 'freezing']):
                return 'Snow'
            elif any(word in x for word in ['rain', 'drizzle', 'shower']):
                return 'Rain'
            elif any(word in x for word in ['fog', 'mist', 'haze']):
                return 'Fog'
            elif 'windy' in x or 'squalls' in x:
                return 'Windy'
            elif any(word in x for word in ['cloud', 'overcast']):
                return 'Clouds'
            elif any(word in x for word in ['clear', 'fair']):
                return 'Clear'
            else:
                return 'Other'

        X['weather_grouped'] = X['Weather_Condition'].apply(weather_group)
        X.drop(columns=['Weather_Condition'], inplace=True)

        return X

class InfrastructureFeatures(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X['low_visibility'] = (X['Visibility(mi)'] < 5).astype(int)
        X['high_humidity'] = (X['Humidity(%)'] > 80).astype(int)
        X['bad_weather'] = (X['low_visibility'] & X['precipitation_event']).astype(int)

        X['infrastructure'] = (
            X['Traffic_Signal'] +
            X['Crossing'] +
            X['Stop']
        )

        X['risk'] = X['Junction']

        return X

class GeoCluster(BaseEstimator, TransformerMixin):

    def __init__(self, n_clusters=15):
        self.n_clusters = n_clusters

    def fit(self, X, y=None):
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.kmeans.fit(X[['Start_Lat','Start_Lng']])
        return self

    def transform(self, X):
        X = X.copy()
        X['geo_cluster'] = self.kmeans.predict(X[['Start_Lat','Start_Lng']])
        return X
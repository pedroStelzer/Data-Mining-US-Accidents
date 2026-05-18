from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from .transformers import MissingValuesHandler, TimestampFeatures, WindDirectionFeatures, WeatherAggregation, InfrastructureFeatures, DropColumns, DurationFeature, GeoCluster

def get_pipeline(feature_flags=None, model=None):
    if feature_flags is None:
        feature_flags = {
            'duration': True,
            'wind': True,
            'weather': True,
            'geo': True,
            'infrastructure': True,
            'drop_columns': True,
        }

    # Pipeline para atributos numéricos
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # Pode ser variado
        ('scaler', StandardScaler())                   # Pode ser variado
    ])

    # Pipeline para atributos categóricos
    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    num_cols = ['Distance(mi)', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)'
                #,'Start_Lat', 'Start_Lng'
                ,'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)'
                , 'duration'
                ]

    cat_cols = ['Timezone', 'Sunrise_Sunset']
    if feature_flags.get('geo', True):
        cat_cols.append('geo_cluster')
    if feature_flags.get('weather', True):
        cat_cols.append('weather_grouped')

    # Transformador de colunas
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ]
    )

    steps = [
        ('missing_fix', MissingValuesHandler()),
    ]

    if feature_flags.get('duration', True):
        steps.append(('duration', DurationFeature()))
    if feature_flags.get('wind', True):
        steps.append(('wind', WindDirectionFeatures()))
    if feature_flags.get('weather', True):
        steps.append(('weather', WeatherAggregation()))
    if feature_flags.get('geo', True):
        steps.append(('geo', GeoCluster()))
    if feature_flags.get('infrastructure', True):
        steps.append(('infrastructure', InfrastructureFeatures()))

    if feature_flags.get('drop_columns', True):
        steps.append(('drop_columns', DropColumns(['End_Lat', 'End_Lng', 'ID', 'Source', 'Description'
            , 'Street', 'City', 'County', 'Country', 'Zipcode', 'Airport_Code'
            , 'Weather_Timestamp', 'Start_Time', 'End_Time'
            , 'Wind_Chill(F)', 'State', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight'
            , 'Turning_Loop', 'Roundabout', 'Bump', 'Traffic_Calming', 'No_Exit', 'Give_Way'
        ])))

    steps.extend([
        ('prep', preprocessor),
    ])

    steps.append(('model', model))

    return Pipeline(steps=steps)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from .transformers import MissingValuesHandler, TimestampFeatures, WindDirectionFeatures, WeatherAggregation, InfrastructureFeatures, DropColumns, DurationFeature, GeoCluster

def get_pipeline():
    # Pipeline para atributos numéricos
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # Pode ser variado
        ('scaler', StandardScaler())                   # Pode ser variado
    ])

    # Pipeline para atributos categóricos
    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Transformador de colunas
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, ['Temperature(F)', 'Humidity(%)']),
            ('cat', categorical_transformer, ['City', 'Weather_Condition'])
        ])

    # Pipeline Final
    full_pipeline = Pipeline(steps=[
        ('missing_fix', MissingValuesHandler()),

        ('start_time', TimestampFeatures(feature='Start_Time')),
        ('end_time', TimestampFeatures(feature='End_Time')),
        ('wheather_timestamp', TimestampFeatures(feature='Weather_Timestamp')),
        ('duration', DurationFeature()),
        ('wind', WindDirectionFeatures()),
        ('weather', WeatherAggregation()),
        ('geo', GeoCluster()),
        ('infrastructure', InfrastructureFeatures()),

        ('drop_columns', DropColumns(['End_Lat', 'End_Lng', 'ID', 'Source', 'Description'
            , 'Street', 'City', 'County', 'Country', 'Zipcode', 'Airport_Code'
            , 'Weather_Timestamp', 'Start_Time', 'End_Time',
            #, 'Wind_Chill(F)', 'State', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight
            #, 'Turning_Loop', 'Roundabout', 'Bump', 'Traffic_Calming', 'No_Exit', 'Give_Way'
        ])),

        ('prep', preprocessor),
        ('model', None) # Deixamos vazio para testar vários modelos
    ])
    
    return full_pipeline
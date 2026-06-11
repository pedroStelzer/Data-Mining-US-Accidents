from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from .transformers import (
    MissingValuesHandler,
    TimestampFeatures,
    WindDirectionFeatures,
    WeatherAggregation,
    InfrastructureFeatures,
    DropColumns,
    DurationFeature,
    GeoCluster,
    PCATransformer,
    SeveritySampler
)


def get_pipeline(
    feature_flags=None,
    model=None,
    use_pca=False,
    pca_n_components=0.95,
    pca_random_state=42,
    balancer_method='none',
    sampler_random_state=42,
    severity_fractions=None
):

    if balancer_method == 'severity_sampling' and severity_fractions is None:
        try:
            from .config import load_config
            config = load_config()
            severity_fractions = config.get("data_balancing", {}).get("severity_sampling_fractions")
        except Exception:
            raise ValueError("severity_sampling ativo, mas as frações não foram encontradas no config.yaml")
    
    if feature_flags is None:
        feature_flags = {
            'duration': True,
            'wind': True,
            'weather': True,
            'geo': True,
            'infrastructure': True,
            'drop_columns': True,
        }

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    num_cols = [
        'Distance(mi)', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)',
        'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)', 'duration'
    ]

    cat_cols = ['Timezone', 'Sunrise_Sunset']
    if feature_flags.get('geo', True):
        cat_cols.append('geo_cluster')
    if feature_flags.get('weather', True):
        cat_cols.append('weather_grouped')

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
        steps.append(('drop_columns', DropColumns([
            'End_Lat', 'End_Lng', 'ID', 'Source', 'Description',
            'Street', 'City', 'County', 'Country', 'Zipcode', 'Airport_Code',
            'Weather_Timestamp', 'Start_Time', 'End_Time',
            'Wind_Chill(F)', 'State', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight',
            'Turning_Loop', 'Roundabout', 'Bump', 'Traffic_Calming', 'No_Exit', 'Give_Way'
        ])))

    steps.extend([
        ('prep', preprocessor),
    ])

    if use_pca:
        steps.append(('pca', PCATransformer(n_components=pca_n_components, random_state=pca_random_state)))

    if balancer_method and balancer_method != 'none':
        if balancer_method == 'oversample':
            sampler = RandomOverSampler(random_state=sampler_random_state)
        elif balancer_method == 'undersample':
            sampler = RandomUnderSampler(random_state=sampler_random_state)
        elif balancer_method == 'smote':
            sampler = SMOTE(random_state=sampler_random_state)
        elif balancer_method == 'severity_sampling':
            sampler = SeveritySampler(fractions=severity_fractions, random_state=sampler_random_state)
        else:
            raise ValueError(f"Método de balanceamento desconhecido: {balancer_method}")

        steps.append(('sampler', sampler))

    steps.append(('model', model))

    return Pipeline(steps=steps)

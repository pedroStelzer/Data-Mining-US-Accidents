import mlflow
import mlflow.sklearn
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from .pipeline import get_pipeline
from .data_ingestion import load_data

def run_experiment():
    X_train, X_test, y_train, y_test = load_data()
    pipeline = get_pipeline()

    # DEFINIR VARIAÇÕES DE PRÉ-PROCESSAMENTO
    # Use o prefixo 'nome_da_etapa__parâmetro'
    param_dist = {
        'geo__n_clusters': [5, 10, 20],
        'prep__num__scaler': [StandardScaler(), RobustScaler(), 'passthrough'],
        'prep__num__imputer__strategy': ['mean', 'median'],
        'model': [RandomForestClassifier()],
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 10, 20]
    }

    mlflow.set_experiment("US_Accidents_Advanced_Search")

    with mlflow.start_run():
        search = RandomizedSearchCV(pipeline, param_dist, n_iter=10, cv=3, scoring='f1_macro')
        search.fit(X_train, y_train)

        # Registar a melhor combinação de TUDO (incluindo pré-processamento)
        mlflow.log_params(search.best_params_)
        mlflow.log_metric("best_f1", search.best_score_)
        
        # Guardar o pipeline campeão como um modelo pronto a usar
        mlflow.sklearn.log_model(search.best_estimator_, "model_pipeline")
        
        print("Experiência concluída com sucesso!")

if __name__ == "__main__":
    run_experiment()
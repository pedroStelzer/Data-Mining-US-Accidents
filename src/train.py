import mlflow
import mlflow.sklearn
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from .pipeline import get_pipeline
from .data_ingestion import load_data

def run_baseline(experiment_name, models_dict, X_train, X_test, y_train, y_test, feature_flags=None):
    """
    Executa uma comparação de baseline entre vários modelos.
    models_dict: Dicionário {'Nome_Modelo': pipeline_ou_estimador}
    feature_flags: Dicionário com flags de feature engineering, ex: {'duration': True}
    """
    mlflow.set_experiment(experiment_name)

    # Finaliza qualquer run ativa para evitar erros
    if mlflow.active_run():
        mlflow.end_run()

    print(f"\n{'='*60}")
    print(f" Iniciando Baseline: {experiment_name}")
    print(f"{'='*60}")

    # Run Pai: Agrupa todos os modelos do baseline
    with mlflow.start_run(run_name="Comparativo_Baseline"):

        for model_name, pipeline in models_dict.items():
            # Run Filha: Cada modelo individual
            with mlflow.start_run(run_name=model_name, nested=True):
                print(f"Treinando: {model_name}...")

                # 1. Treino
                pipeline.fit(X_train, y_train)

                # 2. Predição
                y_pred = pipeline.predict(X_test)

                # 3. Métricas (usando 'macro' por causa do desbalanceamento)
                metrics = {
                    "acc": accuracy_score(y_test, y_pred),
                    "f1_macro": f1_score(y_test, y_pred, average='macro'),
                    "precision_macro": precision_score(y_test, y_pred, average='macro'),
                    "recall_macro": recall_score(y_test, y_pred, average='macro')
                }

                # 4. Registro no MLflow
                # Tenta extrair os parâmetros do modelo se for um pipeline
                if hasattr(pipeline, 'named_steps'):
                    model_step = pipeline.steps[-1][1]
                    mlflow.log_params(model_step.get_params())

                if feature_flags:
                    mlflow.log_params({f"fe_{k}": v for k, v in feature_flags.items()})
                    active_feats = [k for k, v in feature_flags.items() if v]
                    if active_feats:
                        mlflow.set_tag("feature_engineering_steps", ",".join(sorted(active_feats)))

                mlflow.log_metrics(metrics)
                mlflow.set_tag("model_type", model_name)

                # 5. Artefato: Matriz de Confusão
                fig, ax = plt.subplots(figsize=(6, 5))
                ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap='Blues')
                ax.set_title(f"Matriz: {model_name}")
                mlflow.log_figure(fig, f"cm_{model_name}.png")
                plt.close()

                # 6. Salvar Modelo
                mlflow.sklearn.log_model(pipeline, f"model_{model_name}")

                print(f"-> {model_name} finalizado. Accuracy: {metrics['acc']:.4f}. F1: {metrics['f1_macro']:.4f}")

    print("\nBaseline concluído! Verifique o MLflow UI para comparar.")


def run_random_search(
    experiment_name,
    pipeline,
    param_distributions,
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    X_test=None,
    y_test=None,
    feature_flags=None,
    n_iter=20,
    cv=3,
    scoring='accuracy',
    random_state=42,
):
    mlflow.set_experiment(experiment_name)

    if mlflow.active_run():
        mlflow.end_run()

    print(f"\n{'='*60}")
    print(f" Iniciando RandomizedSearch: {experiment_name}")
    print(f"{'='*60}")

    with mlflow.start_run(run_name="RandomizedSearch"):
        if feature_flags:
            mlflow.log_params({f"fe_{k}": v for k, v in feature_flags.items()})
            active_feats = [k for k, v in feature_flags.items() if v]
            if active_feats:
                mlflow.set_tag("feature_engineering_steps", ",".join(sorted(active_feats)))

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            random_state=random_state,
            n_jobs=-1,
            verbose=1,
            return_train_score=False,
        )

        search.fit(X_train, y_train)

        mlflow.log_params(search.best_params_)
        mlflow.log_metric("best_cv_score", search.best_score_)

        if hasattr(search.best_estimator_, 'named_steps'):
            model_step = search.best_estimator_.steps[-1][1]
            mlflow.log_params(model_step.get_params())

        eval_X = X_val if X_val is not None else X_test
        eval_y = y_val if y_val is not None else y_test
        eval_split = 'validation' if X_val is not None else ('test' if X_test is not None else None)

        if eval_X is not None and eval_y is not None:
            y_pred = search.best_estimator_.predict(eval_X)
            eval_metrics = {
                "eval_accuracy": accuracy_score(eval_y, y_pred),
                "eval_f1_macro": f1_score(eval_y, y_pred, average='macro'),
                "eval_precision_macro": precision_score(eval_y, y_pred, average='macro'),
                "eval_recall_macro": recall_score(eval_y, y_pred, average='macro')
            }
            mlflow.log_metrics(eval_metrics)
            if eval_split:
                mlflow.set_tag("evaluation_split", eval_split)

            print(f"Avaliação em {eval_split}: Accuracy={eval_metrics['eval_accuracy']:.4f}, F1={eval_metrics['eval_f1_macro']:.4f}")

        mlflow.sklearn.log_model(search.best_estimator_, "best_model")

        print(f"Melhor resultado CV: {search.best_score_:.4f}")
        print(f"Melhores parâmetros: {search.best_params_}")

    return search
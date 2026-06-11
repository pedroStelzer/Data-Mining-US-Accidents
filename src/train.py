import mlflow
import mlflow.sklearn
import numpy as np
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from .pipeline import get_pipeline
from .data_ingestion import load_data
from src.statistical_tests import paired_wilcoxon_test, format_comparison_report
from .persistence import save_cv_results, plot_train_vs_validation_curves, generate_cv_summary

def run_baseline(
    experiment_name,
    models_dict,
    X_train,
    X_test,
    y_train,
    y_test,
    feature_flags=None,
    data_balance_method='severity_sampling',
    resampling_method='none',
    use_pca=False,
    pca_n_components=None,
    experiment_group='baseline',
):
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
    with mlflow.start_run(run_name=experiment_name):

        for model_name, pipeline in models_dict.items():
            # Gerar run name descritivo: [modelo] | [FE] | [tipo]
            fe_list = ",".join(sorted([k for k, v in feature_flags.items() if v])) if feature_flags else "none"
            run_name = f"{model_name} | FE=[{fe_list}] | baseline"
            
            # Run Filha: Cada modelo individual
            with mlflow.start_run(run_name=run_name, nested=True):
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

                mlflow.set_tag("resampling_method", resampling_method)
                mlflow.set_tag("experiment_group", experiment_group)
                mlflow.set_tag("pca_enabled", str(use_pca).lower())
                if pca_n_components is not None:
                    mlflow.log_param("pca_n_components", pca_n_components)
                mlflow.log_param("resampling_method", resampling_method)
                mlflow.log_param("pca_enabled", use_pca)

                mlflow.log_metrics(metrics)
                mlflow.set_tag("model_type", model_name)
                mlflow.set_tag("run_type", "baseline")
                mlflow.set_tag("model_name", model_name)
                mlflow.set_tag("features_enabled", fe_list)

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
    model_name,
    pipeline,
    param_distributions,
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    X_test=None,
    y_test=None,
    feature_flags=None,
    resampling_method='none',
    use_pca=False,
    pca_n_components=None,
    experiment_group='hyperparameter_search',
    n_iter=20,
    cv=5,
    scoring='f1_macro',
    random_state=42,
):
    mlflow.set_experiment(experiment_name)

    if mlflow.active_run():
        mlflow.end_run()

    print(f"\n{'='*60}")
    print(f" Iniciando RandomizedSearch no Experimento: {experiment_name} | Modelo: {model_name}")
    print(f"{'='*60}")

    fe_list = ",".join(sorted([k for k, v in feature_flags.items() if v])) if feature_flags else "none"
    run_name = f"{model_name} | FE=[{fe_list}] | iter={n_iter}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("run_type", experiment_group)
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("features_enabled", fe_list)
        mlflow.set_tag("n_iterations", n_iter)
        mlflow.set_tag("scoring_metric", scoring)
        mlflow.set_tag("resampling_method", resampling_method)
        mlflow.set_tag("experiment_group", experiment_group)
        mlflow.set_tag("pca_enabled", str(use_pca).lower())
        
        # Determina a quantidade de folds real
        n_splits = cv.get_n_splits() if hasattr(cv, 'get_n_splits') else cv
        mlflow.set_tag("cv_folds", n_splits)

        if pca_n_components is not None:
            mlflow.log_param("pca_n_components", pca_n_components)
        mlflow.log_param("resampling_method", resampling_method)
        mlflow.log_param("pca_enabled", use_pca)

        if feature_flags:
            mlflow.log_params({f"fe_{k}": v for k, v in feature_flags.items()})

        # Executa a busca
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            random_state=random_state,
            n_jobs=-1,
            verbose=1,
            return_train_score=True,
        )

        search.fit(X_train, y_train)

        # Salva os artefatos padrão de curvas e sumarização
        unique_file_identifier = f"{experiment_name}_{model_name}"
        cv_path = save_cv_results(search, unique_file_identifier)
        plot_path = plot_train_vs_validation_curves(search, unique_file_identifier)
        summary_path = generate_cv_summary(search, unique_file_identifier)

        for artifact_path in [cv_path, plot_path, summary_path]:
            if artifact_path is not None:
                try:
                    mlflow.log_artifact(str(artifact_path))
                except Exception:
                    pass

        mlflow.log_params(search.best_params_)
        mlflow.log_metric("best_cv_score", search.best_score_)

        best_idx = search.best_index_
        best_fold_scores = [
            search.cv_results_[f"split{i}_test_score"][best_idx] 
            for i in range(n_splits)
        ]
        
        # Loga individualmente o score de cada fold no MLflow para o modelo atual
        for i, fold_score in enumerate(best_fold_scores):
            mlflow.log_metric(f"cv_fold_{i}_score", fold_score)
        print(f" -> Scores dos {n_splits} folds extraídos e salvos com sucesso.")

        best_idx = search.best_index_
        mean_score = search.cv_results_["mean_test_score"][best_idx]
        std_score = search.cv_results_["std_test_score"][best_idx]
        
        # Print formatado exatamente como a rubrica exige:
        print(f"\n Resultado Final CV -> {scoring} = {mean_score:.3f} ± {std_score:.3f}")
        
        # Logando a média e o desvio separadamente no MLflow
        mlflow.log_metric("best_cv_score_mean", mean_score)
        mlflow.log_metric("best_cv_score_std", std_score)

        # Extração dos scores individuais por fold para o Wilcoxon
        best_fold_scores = [
            search.cv_results_[f"split{i}_test_score"][best_idx] 
            for i in range(n_splits)
        ]
        
        for i, fold_score in enumerate(best_fold_scores):
            mlflow.log_metric(f"cv_fold_{i}_score", fold_score)

        # =========================================================================
        # TESTE DE WILCOXON DIRETO CONTRA O BASELINE VIA MLflow
        # =========================================================================
        if "none" not in experiment_group:
            print(f" -> Buscando o histórico de Baseline para o teste de Wilcoxon...")
            try:
                from mlflow.tracking import MlflowClient
                client = MlflowClient()
                
                # 1. Localiza o experimento de baseline
                baseline_exp_id = 24
                
                if baseline_exp_id:
                    # 2. Busca a Run do Baseline correspondente para este MESMO modelo
                    runs_baseline = client.search_runs(
                        experiment_ids=[baseline_exp_id],
                        filter_string=f"tags.model_name = '{model_name}'"
                    )
                    
                    if runs_baseline:
                        baseline_run = runs_baseline[0]
                        # Extrai os scores salvos nos metadados daquela Run do Baseline
                        baseline_fold_scores = []
                        for i in range(n_splits):
                            val = baseline_run.data.metrics.get(f"cv_fold_{i}_score")
                            if val is not None:
                                baseline_fold_scores.append(val)

                        print(f"\nbaseline_fold_scores -> {baseline_fold_scores}\nbest_fold_scores -> {best_fold_scores}")
                        
                        # 3. Executa o teste pareado se as dimensões baterem perfeitamente
                        if len(baseline_fold_scores) == n_splits:
                            wilcoxon_result = paired_wilcoxon_test(
                                np.array(baseline_fold_scores), 
                                np.array(best_fold_scores)
                            )
                            
                            # Extrai o p-valor dinamicamente (tratando se for objeto ou float direto)
                            p_value = wilcoxon_result["p_value"]
                            
                            print(f"[TESTE ESTATÍSTICO] Wilcoxon Pareado realizado.")
                            print(f"   -> p-valor obtido: {p_value:.5f}")
                            if p_value < 0.05:
                                print("   -> Resultado: Estatisticamente significante (p < 0.05)!")
                            else:
                                print("   -> Resultado: Não há evidência estatística de diferença (p >= 0.05).")

                            # Formata o relatório utilizando a métrica alvo
                            report_txt = format_comparison_report(wilcoxon_result, scoring)
                            mlflow.log_metric("wilcoxon_p_value", p_value)

                            # Salva o arquivo de texto do relatório como artefato da Run atual
                            stat_path = Path(f"artifacts/wilcoxon_report_{model_name}.txt")
                            stat_path.parent.mkdir(parents=True, exist_ok=True)
                            with open(stat_path, "w") as f_out:
                                f_out.write(report_txt)
                            mlflow.log_artifact(str(stat_path))
                        else:
                            print(" ⚠ Falha: O número de folds encontrados no Baseline difere do experimento atual.")
                    else:
                        print(f" ⚠ Baseline Puro para o modelo {model_name} não foi encontrado no MLflow para comparação.")
            except Exception as e:
                print(f" ⚠ Erro ao processar o teste estatístico automático: {e}")
        # =========================================================================

        # Código de avaliação em validação/teste (Mantido original)
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

        mlflow.sklearn.log_model(search.best_estimator_, f"model_{model_name}")
        print(f"Melhor resultado CV: {search.best_score_:.4f}")
        print(f"Melhores parâmetros: {search.best_params_}")

    return search
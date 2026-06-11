import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from pathlib import Path
import json


mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

modelos = [
    "RandomForest",
    "XGBoost",
    "LogisticRegression",
    "DecisionTree",
    "KNN",
    "NaiveBayes",
    "MLP",
    "AdaBoost",
    "Bagging",
    "SVM",
    "Voting",
    "Stacking",
    "LightGBM",
    "LVQ_GLVQ"
]

experiment_ids = [11]

print(f"Processando {len(experiment_ids)} experimentos: {experiment_ids}\n")

for exp_id in experiment_ids:
    try:
        # Obter informações do experimento
        experiment = client.get_experiment(exp_id)
        exp_name = experiment.name.replace("/", "_").replace(" ", "_")
        
        print(f"[Experimento {exp_id}] {experiment.name}")
        
        resultados = []
        
        for model_name in modelos:
            runs = client.search_runs(
                experiment_ids=[exp_id],
                filter_string=f"tags.model_name = '{model_name}'"
            )
            
            for run in runs:
                metrics = run.data.metrics
                params = run.data.params
                
                # Extrair as métricas com 5 casas decimais
                best_cv_score_mean = metrics.get("best_cv_score_mean", None)
                best_cv_score_std = metrics.get("best_cv_score_std", None)
                wilcoxon_p_value = metrics.get("wilcoxon_p_value", None)
                
                # Extrair os 5 fold scores
                fold_scores = []
                for i in range(5):
                    fold_key = f"cv_fold_{i}_score"
                    fold_score = metrics.get(fold_key, None)
                    fold_scores.append(fold_score)
                
                # Armazenar os dados
                row = {
                    "modelo": model_name,
                    "best_cv_score_mean": round(best_cv_score_mean, 5) if best_cv_score_mean else None,
                    "best_cv_score_std": round(best_cv_score_std, 5) if best_cv_score_std else None,
                    "cv_fold_score_0": round(fold_scores[0], 5) if fold_scores[0] else None,
                    "cv_fold_score_1": round(fold_scores[1], 5) if fold_scores[1] else None,
                    "cv_fold_score_2": round(fold_scores[2], 5) if fold_scores[2] else None,
                    "cv_fold_score_3": round(fold_scores[3], 5) if fold_scores[3] else None,
                    "cv_fold_score_4": round(fold_scores[4], 5) if fold_scores[4] else None,
                    "wilcoxon_p_value": round(wilcoxon_p_value, 5) if wilcoxon_p_value else None,
                }
                
                # Adicionar todos os parâmetros
                for param_key, param_value in params.items():
                    # Tentar converter para tipo apropriado
                    try:
                        # Tenta converter para float
                        row[f"param_{param_key}"] = float(param_value)
                    except (ValueError, TypeError):
                        try:
                            # Tenta converter para int
                            row[f"param_{param_key}"] = int(param_value)
                        except (ValueError, TypeError):
                            # Mantém como string
                            row[f"param_{param_key}"] = param_value
                
                resultados.append(row)
        
        # Criar DataFrame e salvar em CSV
        if resultados:
            df = pd.DataFrame(resultados)
            csv_filename = f"{exp_name}_with_params.csv"
            df.to_csv(csv_filename, index=False)
            print(f"  ✓ Dados salvos em '{csv_filename}' ({len(resultados)} modelos)")
            print(f"    Colunas: {len(df.columns)} (métricas + parâmetros)\n")
        else:
            print(f"  ⚠ Nenhum dado encontrado para este experimento\n")
            
    except Exception as e:
        print(f"  ✗ Erro ao processar experimento {exp_id}: {e}\n")

print("✓ Processamento concluído!")

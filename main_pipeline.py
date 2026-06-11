#!/usr/bin/env python3
"""
Script de Orquestração Final - Executa pipeline completo de análise.

Etapas:
1. Treina todos os 10 modelos com busca de hiperparâmetros
2. Compara resultados na validação
3. Seleciona best model
4. Avalia no conjunto de teste (final)
5. Gera relatório e gráficos
6. Testa Wilcoxon pareado entre melhor modelo vs baseline
"""

import argparse
import os
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score

# Local imports
from src.config import load_config
from src.data_ingestion import load_data
from src.run_experiment import run_all_experiments, run_feature_engineering_ablation, run_all_experiments_variations
from src.train import run_baseline, run_random_search
from src.statistical_tests import paired_wilcoxon_test, format_comparison_report
from src.evaluation import evaluate_final_model, compare_multiple_models_on_test
from src.persistence import save_cv_results


def main():
    """Orquestra execução completa do pipeline."""
    config = load_config()

    parser = argparse.ArgumentParser(description="Executa o pipeline completo de US Accidents")

    parser.add_argument(
        "--resampling-method",
        choices=["none", "oversample", "undersample", "smote", "severity_sampling"],
        default=config.get("data_balancing", {}).get("resampling_method", "none"),
        help="Método de balanceamento aplicado no pipeline de treino."
    )
    parser.add_argument(
        "--use-pca",
        action="store_true",
        default=config.get("feature_engineering", {}).get("pca", {}).get("enabled", False),
        help="Habilita PCA como etapa de pré-processamento."
    )
    parser.add_argument(
        "--pca-n-components",
        type=float,
        default=config.get("feature_engineering", {}).get("pca", {}).get("n_components", 0.95),
        help="Número de componentes ou proporção da variância explicada para PCA."
    )
    parser.add_argument(
        "--experiment-group",
        type=str,
        default=config.get("experiment", {}).get("group", "final"),
        help="Nome do grupo de experimentos para tags no MLflow."
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=config.get("experiment", {}).get("mlflow_tracking_uri", "sqlite:///mlflow.db"),
        help="URI de rastreamento do MLflow."
    )
    parser.add_argument(
        "--feature-engineering-ablation",
        action="store_true",
        help="Executa ablação automática de feature engineering em vez do pipeline final."
    )
    args = parser.parse_args()

    print("\n" + "="*80)
    print("PIPELINE COMPLETO - US ACCIDENTS SEVERITY CLASSIFICATION")
    print("="*80 + "\n")

    sample_fraction = config.get("data_ingestion", {}).get("sample_fraction", 1.0)
    print(f"[1/6] Carregando dados puros... (Fração dos dados brutos: {sample_fraction * 100}%)")
    
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    
    # Chamada atualizada conforme a nova lógica de ingestão em cache
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(
        sample_fraction=sample_fraction,
        seed=config.get("experiment", {}).get("seed", 42),
    )

    y_train = y_train - 1
    y_val = y_val - 1
    y_test = y_test - 1

    print(f"  ✓ Treino: {X_train.shape[0]} | Validação: {X_val.shape[0]} | Teste: {X_test.shape[0]}\n")

    if args.feature_engineering_ablation:
        print("[2/6] Executando ablação automática de feature engineering...")
        run_feature_engineering_ablation(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            feature_flags={
                'duration': config.get('feature_engineering', {}).get('duration', True),
                'wind': config.get('feature_engineering', {}).get('wind', True),
                'weather': config.get('feature_engineering', {}).get('weather', True),
                'geo': config.get('feature_engineering', {}).get('geo', True),
                'drop_columns': config.get('feature_engineering', {}).get('drop_columns', True),
                'infrastructure': config.get('feature_engineering', {}).get('infrastructure', True),
            },
            resampling_method=args.resampling_method,
            use_pca=args.use_pca,
            pca_n_components=args.pca_n_components,
            experiment_group=args.experiment_group,
        )
        print("  ✓ Ablação de feature engineering concluída\n")
        return 0

    print("[2/6] Executando busca de hiperparâmetros (10 modelos)...")
    if "baseline" in args.experiment_group:
        print("\n[EXECUÇÃO] Iniciando Passo 1: Baseline Puro (Hiperparâmetros padrão e Sem FE avançada)...")
        
        # Desliga todas as transformações avançadas de Feature Engineering
        feature_flags_baseline = {
            'duration': False,
            'wind': False,
            'weather': False,
            'geo': True,
            'infrastructure': False,
            'drop_columns': True  # Mantém apenas para limpar IDs e textos brutos
        }
        
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neural_network import MLPClassifier
        from sklearn.svm import SVC
        from src.pipeline import get_pipeline
        from src.train import run_baseline

        # Função auxiliar para criar o pipeline básico fixando as flags de baseline
        def create_baseline_pipe(model_obj):
            return get_pipeline(
                feature_flags=feature_flags_baseline,
                model=model_obj,
                use_pca=args.use_pca,
                pca_n_components=args.pca_n_components,
                balancer_method=args.resampling_method,
                severity_fractions=config.get("data_balancing", {}).get("severity_sampling_fractions")
            )

        # Dicionário de modelos com os parâmetros padrão (default) das bibliotecas
        # --- SUPER MONKEY PATCH DE COMPATIBILIDADE (VALIDAÇÃO + PREDIÇÃO) ---
        import sklearn.base
        import sklearn.utils
        from sklearn.utils import check_X_y, check_array

        # 1. Patch para o momento do Treino (.fit)
        def _compat_validate_data(self, X, y=None, ensure_2d=True, dtype="numeric", accept_sparse=False, **kwargs):
            kwargs.pop('force_all_finite', None)
            kwargs.pop('allow_nd', None)
            kwargs.pop('cast_to_ndarray', None)
            if y is not None:
                X_validated, y_validated = check_X_y(X, y, accept_sparse=accept_sparse, dtype=dtype, ensure_2d=ensure_2d, **kwargs)
                self.n_features_in_ = X_validated.shape[1]  # Garante as colunas no modelo
                return X_validated, y_validated
            return X

        if not hasattr(sklearn.base.BaseEstimator, "_validate_data"):
            sklearn.base.BaseEstimator._validate_data = _compat_validate_data


        # 2. Patch para o momento da Predição (.predict / decision_function)
        # Guarda uma referência da função original do Scikit-Learn
        _original_check_array = sklearn.utils.check_array

        def _compat_check_array(array, *args, **kwargs):
            # Se o sklvq passar o argumento antigo, nós traduzimos ou removemos
            if 'force_all_finite' in kwargs:
                # Nas versões novas do sklearn, o comportamento padrão já lida com isso, podemos remover
                kwargs.pop('force_all_finite')
            return _original_check_array(array, *args, **kwargs)

        # Substitui a função globalmente no módulo do Sklearn para o sklvq usar a nossa modificada
        sklearn.utils.check_array = _compat_check_array
        sklearn.utils.validation.check_array = _compat_check_array
        # ---------------------------------------------------------------------
        # 1. Inicializa o dicionário VAZIO
        baselines = {}

        estimators = [
            ('lr', LogisticRegression(max_iter=1000)),
            ('dt', DecisionTreeClassifier(max_depth=5)),
            ('rf', RandomForestClassifier(n_estimators=50))
        ]

        # 2. Tenta colocar o GLVQ no TOPO da fila para testar primeiro
        try:
            from sklvq import GLVQ
            # Nota: use 'squared-euclidean' com hífen se der erro de string
            baselines['LVQ_GLVQ'] = create_baseline_pipe(GLVQ(distance_type='squared-euclidean', random_state=42))
        except ImportError:
            print("  ⚠ GLVQ não encontrado, pulando do baseline.")

        # 3. Alimenta o restante dos modelos padrões (eles vão entrar DEPOIS do GLVQ)
        baselines.update({
            'LogisticRegression': create_baseline_pipe(LogisticRegression(max_iter=1000)),
            'DecisionTree': create_baseline_pipe(DecisionTreeClassifier(max_depth=5)),
            'RandomForest': create_baseline_pipe(RandomForestClassifier(n_estimators=50, n_jobs=-1)),
            'KNN': create_baseline_pipe(KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
            'NaiveBayes': create_baseline_pipe(GaussianNB()),
            'MLP': create_baseline_pipe(MLPClassifier(max_iter=300)),
            'AdaBoost': create_baseline_pipe(AdaBoostClassifier()),
            'Bagging': create_baseline_pipe(BaggingClassifier()),
            'Voting': create_baseline_pipe(VotingClassifier(estimators=estimators)),
            'Stacking': create_baseline_pipe(StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())),
            'SVM': create_baseline_pipe(SVC(kernel='linear', max_iter=1000, probability=True))
        })

        #4. Adiciona os opcionais dinamicamente no final
        try:
            from xgboost import XGBClassifier
            baselines['XGBoost'] = create_baseline_pipe(XGBClassifier(n_jobs=-1))
        except ImportError:
            print("  ⚠ XGBoost não encontrado, pulando do baseline.")

        try:
            from lightgbm import LGBMClassifier
            baselines['LightGBM'] = create_baseline_pipe(LGBMClassifier(n_jobs=-1))
        except ImportError:
            print("  ⚠ LightGBM não encontrado, pulando do baseline.")

        # Dispara o treino linear direto (método .fit() puro, sem RandomizedSearchCV)
        run_baseline(
            experiment_name='US_Accidents_PCA',
            models_dict=baselines,
            X_train=X_train,
            X_test=X_val,  # Avalia no conjunto de validação para manter o alinhamento
            y_train=y_train,
            y_test=y_val,
            feature_flags=feature_flags_baseline,
            resampling_method=args.resampling_method,
            use_pca=args.use_pca,
            pca_n_components=args.pca_n_components,
            experiment_group=args.experiment_group,
        )
        print("  ✓ Todos os 12 modelos de Baseline Puro executados com sucesso!\n")

    elif("balancing_method" in args.experiment_group or "pca" in args.experiment_group):

        print(f"\n[EXECUÇÃO] Iniciando experimentos (Grupo: {args.experiment_group})...")
        
        feature_flags_full = {
            'duration': config.get('feature_engineering', {}).get('duration', True),
            'wind': config.get('feature_engineering', {}).get('wind', True),
            'weather': config.get('feature_engineering', {}).get('weather', True),
            'geo': config.get('feature_engineering', {}).get('geo', True),
            'drop_columns': config.get('feature_engineering', {}).get('drop_columns', True),
            'infrastructure': config.get('feature_engineering', {}).get('infrastructure', True),
        }
        
        run_all_experiments_variations(
            X_train=X_train, X_val=X_val, X_test=X_test,
            y_train=y_train, y_val=y_val, y_test=y_test,
            feature_flags=feature_flags_full,
            resampling_method=args.resampling_method,
            use_pca=args.use_pca,
            pca_n_components=args.pca_n_components,
            experiment_group=args.experiment_group,
        )
        print("  ✓ Rodada concluída!\n")
    # 💡 SE FOR O PASSO 5: Executa o Estudo de Ablação Automática
    elif getattr(args, 'feature_engineering_ablation', False) or args.experiment_group == "fe_ablation":
        print(f"\n[EXECUÇÃO] Iniciando Passo 5: Estudo de Ablação Automática ({args.experiment_group})...")
        
        # Recupera as configurações completas do config.yaml
        feature_flags_full = {
            'duration': config.get('feature_engineering', {}).get('duration', True),
            'wind': config.get('feature_engineering', {}).get('wind', True),
            'weather': config.get('feature_engineering', {}).get('weather', True),
            'geo': config.get('feature_engineering', {}).get('geo', True),
            'drop_columns': config.get('feature_engineering', {}).get('drop_columns', True),
            'infrastructure': config.get('feature_engineering', {}).get('infrastructure', True),
        }
        
        run_feature_engineering_ablation(
            X_train=X_train, X_val=X_val, X_test=X_test,
            y_train=y_train, y_val=y_val, y_test=y_test,
            feature_flags=feature_flags_full,
            resampling_method=args.resampling_method,
            use_pca=args.use_pca,
            pca_n_components=args.pca_n_components,
            experiment_group=args.experiment_group,
            include_all_disabled=True
        )
        print("  ✓ Experimentos de ablação concluídos!\n")

    # 💡 SE FOR QUALQUER OUTRO PASSO (2, 3, 4, 6): Executa HPTuning Completo (RandomSearch)
    else:
        print(f"\n[EXECUÇÃO] Iniciando experimentos com Busca de Hiperparâmetros (Grupo: {args.experiment_group})...")
        
        feature_flags_full = {
            'duration': config.get('feature_engineering', {}).get('duration', True),
            'wind': config.get('feature_engineering', {}).get('wind', True),
            'weather': config.get('feature_engineering', {}).get('weather', True),
            'geo': config.get('feature_engineering', {}).get('geo', True),
            'drop_columns': config.get('feature_engineering', {}).get('drop_columns', True),
            'infrastructure': config.get('feature_engineering', {}).get('infrastructure', True),
        }
        
        run_all_experiments(
            X_train=X_train, X_val=X_val, X_test=X_test,
            y_train=y_train, y_val=y_val, y_test=y_test,
            feature_flags=feature_flags_full,
            resampling_method=args.resampling_method,
            use_pca=args.use_pca,
            pca_n_components=args.pca_n_components,
            experiment_group=args.experiment_group,
        )
        print("  ✓ Rodada de busca de hiperparâmetros concluída!\n")

    """
    print("[3/6] Selecionando melhor modelo...")
    client = mlflow.tracking.MlflowClient()

    all_experiments = client.search_experiments()
    
    experiment_ids = [
        exp.experiment_id for exp in all_experiments 
        if "US_Accidents_" in exp.name
    ]

    if not experiment_ids:
        print(" Nenhum experimento 'US_Accidents_' encontrado no MLflow!")
        return 1

    print(f"  -> Buscando o melhor modelo entre {len(experiment_ids)} experimentos analisados...")

    # 1. Baixa todas as runs do experimento (Pais e Filhas)
    # Filtramos para trazer apenas runs que possuem o campo 'model_name' preenchido nas tags (isso descarta a Run Pai vazia)
    runs = client.search_runs(
        experiment_ids=experiment_ids,
        filter_string="tags.model_name LIKE '%'"
    )

    if runs:
        # 2. Ordena de forma híbrida no Python para capturar 'f1_macro' (Baseline) ou 'eval_f1_macro' (Tuning)
        def get_best_f1(run):
            metrics_dict = run.data.metrics
            # Pega o maior valor entre as duas chaves se ambas existirem, ou o que estiver disponível
            return max(metrics_dict.get('best_cv_score', 0.0), metrics_dict.get('f1_macro', 0.0))

        # Ordena a lista de runs colocando o maior F1 no topo
        sorted_runs = sorted(runs, key=get_best_f1, reverse=True)
        
        best_run = sorted_runs[0]
        best_metrics = best_run.data.metrics
        best_model_name = best_run.data.tags.get("model_name", "Unknown")

        print(f"   Best Model encontrado: {best_model_name}")
        
        # 3. Exibe o F1 correto baseado no que foi capturado
        f1_val = best_metrics.get('eval_f1_macro') or best_metrics.get('f1_macro')
        if f1_val is not None:
            print(f"   F1-Macro Escolhido: {f1_val:.4f}\n")
        else:
            print(f"   F1-Macro Escolhido: N/A\n")

        # Mantém a lógica de pastas de artefatos intacta
        run_type = best_run.data.tags.get("run_type", "baseline")
        if run_type == "baseline":
            artifact_folder = f"model_{best_model_name}"
        else:
            artifact_folder = f"model_{best_model_name}"

        model_path = client.download_artifacts(best_run.info.run_id, artifact_folder, dst_path="artifacts")
        
        sys.path.insert(0, model_path)
        best_model = pickle.load(open(f"{model_path}/model.pkl", "rb"))
    else:
        print(" Nenhum modelo encontrado nas runs dos experimentos!")
        return 1

    print("[4/6] Avaliação final no conjunto de teste...")
    final_metrics = evaluate_final_model(
        best_model, X_test, y_test,
        model_name=best_model_name,
        output_dir="final_evaluation"
    )
    print(f"  ✓ Gráficos gerados: ROC, Precision-Recall, Matriz de Confusão\n")

    print("[5/6] Teste estatístico (Wilcoxon: Best vs Baseline)...")

    # 1. Localiza o ID do experimento de Baselines Puros de forma dinâmica e flexível
    baseline_exp_id = None
    try:
        all_experiments = client.search_experiments()
        for exp in all_experiments:
            # Busca flexível: precisa conter 'US_Accidents' e a palavra 'baseline' no nome do grupo
            if "US_Accidents" in exp.name and "baseline" in exp.name.lower():
                baseline_exp_id = exp.experiment_id
                print(f"  -> Experimento de Baseline localizado dinamicamente: '{exp.name}' (ID: {baseline_exp_id})")
                break
    except Exception as e:
        print(f"  ⚠ Erro ao buscar experimentos no MLflow: {e}")
        baseline_exp_id = None

    baseline_run = None

    if not baseline_exp_id:
        print("Experimento contendo 'US_Accidents' e 'baseline' não foi localizado no MLflow.")
        print("Certifique-se de rodar o Passo 1 do plano lógico antes de prosseguir.\n")
    else:
        # 2. Busca o baseline PURO do MESMO modelo que foi eleito o melhor no passo [3/6]
        print(f"  -> Buscando o baseline puro correspondente para o modelo: {best_model_name}")
        
        runs_baseline = client.search_runs(
            experiment_ids=[baseline_exp_id],
            filter_string=f"tags.model_name = '{best_model_name}' and tags.run_type = 'baseline'"
        )
        
        if runs_baseline:
            baseline_run = runs_baseline[0]
        else:
            # Fallback: Se não achar o mesmo modelo, busca o marco zero clássico (LightGBM)
            print(f" Baseline puro de {best_model_name} não encontrado. Buscando LightGBM como fallback...")
            runs_fallback = client.search_runs(
                experiment_ids=[baseline_exp_id],
                filter_string="tags.model_name = 'LightGBM' and tags.run_type = 'baseline'"
            )
            if runs_fallback:
                baseline_run = runs_fallback[0]

    # 3. Execução do teste se o modelo de referência foi encontrado
    if baseline_run:
        actual_baseline_name = baseline_run.data.tags.get("model_name", "Baseline")
        print(f"  ✓ Baseline Puro localizado: {actual_baseline_name} (Run ID: {baseline_run.info.run_id})")
        print(f"  -> Baixando o arquivo do modelo para validação cruzada...")
        
        baseline_folder = f"model_{actual_baseline_name}"
        baseline_path = client.download_artifacts(baseline_run.info.run_id, baseline_folder, dst_path="artifacts")
        
        with open(f"{baseline_path}/model.pkl", "rb") as f:
            baseline_model = pickle.load(f)

        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        baseline_scores = []
        best_scores = []

        # 💡 ADAPTAÇÃO DINÂMICA DE SEGURANÇA CONTRA OOM
        resampling_actual = args.resampling_method
        
        if resampling_actual == "severity_sampling":
            print(f"  -> [Aviso] {resampling_actual} ativo. Reduzindo dados para 5% apenas no teste Wilcoxon (Proteção de RAM)...")
            X_wilcoxon = X_train.sample(frac=0.05, random_state=42) if hasattr(X_train, 'sample') else X_train[:200000]
            y_wilcoxon = y_train.iloc[X_wilcoxon.index] if hasattr(y_train, 'iloc') else y_train[X_wilcoxon.index]
        else:
            print(f"  -> [Info] {resampling_actual} ativo (dados já reduzidos via config). Usando 100% para o Wilcoxon...")
            X_wilcoxon = X_train
            y_wilcoxon = y_train

        print(f"  -> Rodando K-Fold (5 splits): {actual_baseline_name} vs {best_model_name} (Tunado)...")
        for train_idx, val_idx in skf.split(X_wilcoxon, y_wilcoxon):
            X_train_fold, X_val_fold = X_wilcoxon.iloc[train_idx], X_wilcoxon.iloc[val_idx]
            
            # Indexação segura compatível com Pandas Series e NumPy Arrays (evita quebra por falta de .iloc)
            y_train_fold = y_wilcoxon.iloc[train_idx] if hasattr(y_wilcoxon, 'iloc') else y_wilcoxon[train_idx]
            y_val_fold = y_wilcoxon.iloc[val_idx] if hasattr(y_wilcoxon, 'iloc') else y_wilcoxon[val_idx]

            # Treina e avalia o baseline puro no fold
            baseline_model.fit(X_train_fold, y_train_fold)
            baseline_pred = baseline_model.predict(X_val_fold)
            baseline_acc = accuracy_score(y_val_fold, baseline_pred)
            baseline_scores.append(baseline_acc)

            # Treina e avalia o melhor modelo tunado no fold
            best_model.fit(X_train_fold, y_train_fold)
            best_pred = best_model.predict(X_val_fold)
            best_acc = accuracy_score(y_val_fold, best_pred)
            best_scores.append(best_acc)

        # Aplica o teste Wilcoxon emparelhado
        wilcoxon_result = paired_wilcoxon_test(np.array(baseline_scores), np.array(best_scores))
        report = format_comparison_report(wilcoxon_result, "Accuracy")
        print(f"  {report}\n")
    else:
        print("Nenhum modelo de baseline válido foi localizado no experimento especificado. Pulando Wilcoxon.\n")

    print("[6/6] Gerando relatório final...")

    group_suffix = args.experiment_group if args.experiment_group else "execution"
    report_path = Path(f"final_evaluation/REPORT_{group_suffix}.txt")
    
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RELATÓRIO FINAL - US ACCIDENTS SEVERITY CLASSIFICATION\n")
        f.write("="*80 + "\n\n")

        f.write(f"MELHOR MODELO: {best_model_name}\n\n")

        f.write("MÉTRICAS NO CONJUNTO DE TESTE:\n")
        f.write(f"  Accuracy: {final_metrics['accuracy']:.4f}\n")
        f.write(f"  Balanced Accuracy: {final_metrics['balanced_accuracy']:.4f}\n")
        f.write(f"  F1-macro: {final_metrics['f1_macro']:.4f}\n")
        f.write(f"  F1-weighted: {final_metrics['f1_weighted']:.4f}\n\n")

        f.write("ARTEFATOS GERADOS:\n")
        f.write(f"  - Matriz de Confusão: {final_metrics.get('confusion_matrix_path', 'N/A')}\n")
        f.write(f"  - Curva ROC: {final_metrics.get('roc_curve_path', 'N/A')}\n")
        f.write(f"  - Curva Precision-Recall: {final_metrics.get('pr_curve_path', 'N/A')}\n")
        f.write(f"  - CV Results: cv_results_logs/{best_model_name}_cv_results.csv\n")
        f.write(f"  - Sumário CV: cv_results_logs/{best_model_name}_summary.txt\n")
        f.write("\n")

        f.write("RECOMENDAÇÕES:\n")
        #  Validação alterada para Acurácia com o limite de 0.74
        if final_metrics['accuracy'] >= 0.74:
            f.write("  ✓ Modelo aprovado para deployment em produção\n")
        else:
            f.write("  ⚠ Modelo não atinge Accuracy >= 0.74; recomenda-se mais experimentos\n")
        f.write("  - Monitorar performance em dados novos (concept drift)\n")
        f.write("  - Retreinar a cada 3 meses com dados históricos recentes\n")
        f.write("  - Implementar logs de predição para auditoria\n")

    print(f"  ✓ Relatório salvo: {report_path}\n")

    print("="*80)
    print("PIPELINE COMPLETO FINALIZADO COM SUCESSO!")
    print("="*80)
    print("\nArtefatos gerados em:")
    print("  - final_evaluation/")
    print("  - cv_results_logs/")
    print("  - plots_hyperparameter/")
    print("  - MLflow UI: http://localhost:5000\n")
    """
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

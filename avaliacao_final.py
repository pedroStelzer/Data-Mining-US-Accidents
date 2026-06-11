import os
import sys
import pickle
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from src.data_ingestion import load_data
from mlflow.tracking import MlflowClient

# Imports do Scikit-Learn
from sklearn.metrics import (
    accuracy_score, 
    balanced_accuracy_score, 
    f1_score, 
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc, 
    precision_recall_curve
)
from sklearn.preprocessing import label_binarize

def evaluate_final_model(model, X_test, y_test, model_name="Best_Model", output_dir="final_evaluation"):
    """
    Avalia modelo no conjunto de teste final.
    Gera: matriz de confusão, relatório de classificação, métricas.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Predições
    y_pred = model.predict(X_test)
    
    # Possibilidades (para ROC/PR)
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)
    else:
        y_pred_proba = None
    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    metrics = {
        "model_name": model_name,
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "n_test_samples": len(y_test),
    }
    
    # Relatório de classificação
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    metrics["classification_report"] = report
    
    print(f"\n{'='*70}")
    print(f"AVALIAÇÃO FINAL NO TESTE: {model_name}")
    print(f"{'='*70}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"F1-macro: {f1_macro:.4f}")
    print(f"F1-weighted: {f1_weighted:.4f}")
    print(f"\nRelatório de Classificação:\n")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Gráfico 1: Matriz de Confusão Normalizada
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=ax,
                cbar_kws={'label': 'Proporção'}, square=True)
    ax.set_title(f'Matriz de Confusão Normalizada - {model_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predito', fontsize=12)
    ax.set_ylabel('Real', fontsize=12)
    
    cm_path = Path(output_dir) / f"{model_name}_confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    metrics["confusion_matrix_path"] = str(cm_path)
    print(f"Matriz de confusão salva: {cm_path}")
    
    # Gráficos 2 & 3: ROC e PR (multiclasse / binário adaptado)
    if y_pred_proba is not None:
        unique_classes = np.unique(y_test)
        n_classes = len(unique_classes)
        
        # Binarizar labels (one-vs-rest)
        y_test_bin = label_binarize(y_test, classes=unique_classes)
        
        # CORREÇÃO CRÍTICA: Se n_classes == 2, label_binarize retorna formato shape (N, 1). 
        # Convertemos para shape (N, 2) para que o laço de repetição funcione uniformemente.
        if n_classes == 2:
            y_test_bin = np.hstack([1 - y_test_bin, y_test_bin])
        
        # ROC Curve
        fpr_dict = {}
        tpr_dict = {}
        roc_auc_dict = {}
        
        for i, class_label in enumerate(unique_classes):
            fpr_dict[i], tpr_dict[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
        
        # PR Curve
        precision_dict = {}
        recall_dict = {}
        pr_auc_dict = {}
        
        for i, class_label in enumerate(unique_classes):
            precision_dict[i], recall_dict[i], _ = precision_recall_curve(
                y_test_bin[:, i], y_pred_proba[:, i]
            )
            pr_auc_dict[i] = auc(recall_dict[i], precision_dict[i])
        
        # Plotar ROC
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
        
        for i, class_label in enumerate(unique_classes):
            ax.plot(fpr_dict[i], tpr_dict[i], color=colors[i], lw=2,
                   label=f'Classe {class_label} (AUC={roc_auc_dict[i]:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        ax.set_xlabel('Taxa de Falso Positivo', fontsize=12)
        ax.set_ylabel('Taxa de Verdadeiro Positivo', fontsize=12)
        ax.set_title(f'Curva ROC - {model_name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3)
        
        roc_path = Path(output_dir) / f"{model_name}_roc_curve.png"
        fig.tight_layout()
        fig.savefig(roc_path, dpi=150)
        plt.close(fig)
        metrics["roc_curve_path"] = str(roc_path)
        print(f"✓ Curva ROC salva: {roc_path}")
        
        # Plotar PR
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for i, class_label in enumerate(unique_classes):
            ax.plot(recall_dict[i], precision_dict[i], color=colors[i], lw=2,
                   label=f'Classe {class_label} (AUC={pr_auc_dict[i]:.3f})')
        
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precisão', fontsize=12)
        ax.set_title(f'Curva Precision-Recall - {model_name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        pr_path = Path(output_dir) / f"{model_name}_pr_curve.png"
        fig.tight_layout()
        fig.savefig(pr_path, dpi=150)
        plt.close(fig)
        metrics["pr_curve_path"] = str(pr_path)
        print(f"Curva PR salva: {pr_path}")
    
    # Salvar métricas em JSON
    metrics_json_path = Path(output_dir) / f"{model_name}_metrics.json"
    metrics_to_save = metrics.copy()
    metrics_to_save.pop("classification_report", None)
    
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    
    print(f"Métricas salvas: {metrics_json_path}")
    print(f"{'='*70}\n")
    
    return metrics


def execute_full_pipeline(X_test, y_test):
    """
    Função principal que encapsula a busca do modelo no MLflow 
    e dispara a avaliação final do modelo.
    """
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()
    all_experiments = client.search_experiments()

    experiment_ids = [
        exp.experiment_id for exp in all_experiments 
        if "US_Accidents_" in exp.name
    ]

    if not experiment_ids:
        print("Nenhum experimento 'US_Accidents_' encontrado no MLflow!")
        return 1

    print(f"-> Buscando o melhor modelo entre {len(experiment_ids)} experimentos analisados...")

    # 1. Baixa todas as runs do experimento (Pais e Filhas)
    runs = client.search_runs(
        experiment_ids=experiment_ids,
        filter_string="tags.model_name LIKE '%'"
    )

    if runs:
        # 2. Ordena de forma híbrida no Python capturando 'best_cv_score' ou 'f1_macro'
        def get_best_f1(run):
            metrics_dict = run.data.metrics
            return max(metrics_dict.get('best_cv_score', 0.0), metrics_dict.get('f1_macro', 0.0))

        # Ordena a lista de runs colocando o maior F1 no topo
        sorted_runs = sorted(runs, key=get_best_f1, reverse=True)
        
        best_run = sorted_runs[0]
        best_metrics = best_run.data.metrics
        best_model_name = best_run.data.tags.get("model_name", "Unknown")

        print(f"Best Model encontrado: {best_model_name}")
        
        # 3. Exibe o F1 correto baseado no que foi capturado
        f1_val = best_metrics.get('eval_f1_macro') or best_metrics.get('f1_macro') or best_metrics.get('best_cv_score')
        if f1_val is not None:
            print(f"F1-Macro Escolhido: {f1_val:.4f}\n")
        else:
            print(f"F1-Macro Escolhido: N/A\n")

        # Mantém a lógica de pastas de artefatos intacta
        artifact_folder = f"model_{best_model_name}"

        model_path = client.download_artifacts(best_run.info.run_id, artifact_folder, dst_path="artifacts")
        
        sys.path.insert(0, model_path)
        
        with open(f"{model_path}/model.pkl", "rb") as f:
            best_model = pickle.load(f)
    else:
        print("Nenhum modelo encontrado nas runs dos experimentos!")
        return 1

    final_metrics = evaluate_final_model(
        best_model, X_test, y_test,
        model_name=best_model_name,
        output_dir="final_evaluation"
    )
    print(f"Gráficos gerados com sucesso: ROC, Precision-Recall, Matriz de Confusão\n")
    return 0

if __name__ == "__main__":

    X_train, X_val, X_test, y_train, y_val, y_test = load_data(
        sample_fraction=0.05,
        seed=42,
    )
    status = execute_full_pipeline(X_test, y_test-1)
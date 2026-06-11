"""
Avaliação final no conjunto de teste e geração de gráficos (ROC, PR, matriz de confusão).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, f1_score, accuracy_score, balanced_accuracy_score,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import label_binarize
import seaborn as sns
from pathlib import Path


def evaluate_final_model(model, X_test, y_test, model_name="Best_Model", output_dir="final_evaluation"):
    """
    Avalia modelo no conjunto de teste final.
    Gera: matriz de confusão, relatório de classificação, métricas.
    
    Parâmetros:
        model: modelo treinado (Pipeline ou estimador)
        X_test: features do teste
        y_test: labels do teste
        model_name: nome do modelo
        output_dir: diretório para salvar resultados
    
    Retorna:
        dict com métricas e paths dos gráficos
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
    
    # Gráficos 2 & 3: ROC e PR (multiclasse)
    if y_pred_proba is not None:
        unique_classes = np.unique(y_test)
        n_classes = len(unique_classes)
        
        # Binarizar labels (one-vs-rest)
        y_test_bin = label_binarize(y_test, classes=unique_classes)
        
        # ROC Curve
        fpr_dict = {}
        tpr_dict = {}
        roc_auc_dict = {}
        
        for i, class_label in enumerate(unique_classes):
            if n_classes == 2:
                fpr_dict[i], tpr_dict[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            else:
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
        ax.set_title(f'Curva ROC (Multiclasse) - {model_name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3)
        
        roc_path = Path(output_dir) / f"{model_name}_roc_curve.png"
        fig.tight_layout()
        fig.savefig(roc_path, dpi=150)
        plt.close(fig)
        metrics["roc_curve_path"] = str(roc_path)
        print(f"Curva ROC salva: {roc_path}")
        
        # Plotar PR
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for i, class_label in enumerate(unique_classes):
            ax.plot(recall_dict[i], precision_dict[i], color=colors[i], lw=2,
                   label=f'Classe {class_label} (AUC={pr_auc_dict[i]:.3f})')
        
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precisão', fontsize=12)
        ax.set_title(f'Curva Precision-Recall (Multiclasse) - {model_name}', fontsize=14, fontweight='bold')
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
    import json
    metrics_json_path = Path(output_dir) / f"{model_name}_metrics.json"
   
    metrics_to_save = metrics.copy()
    metrics_to_save.pop("classification_report", None)
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    
    print(f"Métricas salvas: {metrics_json_path}")
    print(f"{'='*70}\n")
    
    return metrics


def compare_multiple_models_on_test(models_dict, X_test, y_test, output_dir="final_evaluation"):
    """
    Avalia múltiplos modelos no teste e compara suas métricas.
    
    Parâmetros:
        models_dict: {nome_modelo: modelo_treinado}
        X_test: features do teste
        y_test: labels do teste
        output_dir: diretório para salvar resultados
    
    Retorna:
        DataFrame com comparação de modelos
    """
    results = []
    
    for model_name, model in models_dict.items():
        metrics = evaluate_final_model(model, X_test, y_test, model_name, output_dir)
        results.append(metrics)
    
    # Consolidar em DataFrame
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df[['model_name', 'accuracy', 'balanced_accuracy', 'f1_macro', 'f1_weighted']]
    
    # Salvar comparação
    comparison_path = Path(output_dir) / "models_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Comparação de modelos salva: {comparison_path}\n")
    print(comparison_df.to_string(index=False))
    
    return comparison_df


if __name__ == "__main__":
    print("Módulo de avaliação final importado com sucesso.")
    print("Use: evaluate_final_model, compare_multiple_models_on_test")

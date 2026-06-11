"""
Funções para persistência de resultados de busca de hiperparâmetros e geração de gráficos.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def save_cv_results(search, experiment_name, output_dir="cv_results_logs"):
    """
    Salva cv_results_ de RandomizedSearchCV em CSV.
    
    Parâmetros:
        search: RandomizedSearchCV objeto após fit()
        experiment_name: nome do experimento
        output_dir: diretório para salvar CSVs
    
    Retorna:
        Path do arquivo salvo
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Converter para DataFrame
    cv_results_df = pd.DataFrame(search.cv_results_)
    
    # Salvar CSV
    output_path = Path(output_dir) / f"{experiment_name}_cv_results.csv"
    cv_results_df.to_csv(output_path, index=False)
    
    print(f"✓ CV results salvo: {output_path}")
    return output_path


def plot_train_vs_validation_curves(search, experiment_name, output_dir="plots_hyperparameter"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cv_results = search.cv_results_
    
    if 'mean_train_score' not in cv_results:
        print("⚠ Alerta: 'mean_train_score' não encontrado em cv_results_.")
        return None
    
    mean_train_score = cv_results['mean_train_score']
    mean_test_score = cv_results['mean_test_score']
    std_test_score = cv_results['std_test_score']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(mean_train_score))
    
    ax.plot(x, mean_train_score, 'o-', label='Treino', linewidth=2, markersize=6, alpha=0.7)
    ax.plot(x, mean_test_score, 's-', label='Validação', linewidth=2, markersize=6, alpha=0.7)
    
    ax.fill_between(x, 
                     mean_test_score - std_test_score,
                     mean_test_score + std_test_score,
                     alpha=0.2, label='±1 std (Validação)')
    
    ax.set_xlabel('Iteração de Busca (Configuração)', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Histórico de Busca: Treino vs Validação - {experiment_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    output_path = Path(output_dir) / f"{experiment_name}_train_vs_val.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    
    print(f"✓ Gráfico salvo: {output_path}")
    return output_path

def plot_hyperparameter_impact(search, param_name, experiment_name, output_dir="plots_hyperparameter"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cv_results = search.cv_results_
    
    param_key = f'param_{param_name}'
    if param_key not in cv_results:
        print(f"Parâmetro {param_name} não encontrado nos resultados.")
        return None
    
    # 💡 Convertido para lista/numpy para evitar problemas de tipo do MaskedArray do sklearn
    params = np.array(cv_results[param_key].data, dtype=float) 
    test_scores = cv_results['mean_test_score']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    scatter = ax.scatter(params, test_scores, alpha=0.6, s=100, c=test_scores, cmap='viridis')
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel('Score de Validação (Média)', fontsize=12)
    ax.set_title(f'Impacto de {param_name} - {experiment_name}', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Score', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    output_path = Path(output_dir) / f"{experiment_name}_param_{param_name}.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    
    print(f"Gráfico salvo: {output_path}")
    return output_path

def generate_cv_summary(search, experiment_name, output_dir="cv_results_logs"):
    """
    Gera sumário textual dos melhores resultados.
    
    Parâmetros:
        search: RandomizedSearchCV objeto após fit()
        experiment_name: nome do experimento
        output_dir: diretório para salvar sumário
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    cv_results = search.cv_results_
    best_idx = search.best_index_
    
    summary = f"""
{'='*70}
RESUMO DE BUSCA DE HIPERPARÂMETROS
{'='*70}

Experimento: {experiment_name}
Total de iterações: {len(cv_results['mean_test_score'])}

MELHOR CONFIGURAÇÃO (índice {best_idx}):
  CV Score: {search.best_score_:.4f}
  Parâmetros: {search.best_params_}

TOP 5 CONFIGURAÇÕES:
"""
    
    # Ordenar por score
    scores = cv_results['mean_test_score']
    top_5_indices = np.argsort(scores)[-5:][::-1]
    
    for rank, idx in enumerate(top_5_indices, 1):
        summary += f"\n  {rank}. Score={scores[idx]:.4f}"
        # Mostrar parâmetros da configuração
        for key in search.best_params_.keys():
            param_key = f'param_{key}'
            if param_key in cv_results:
                summary += f"\n     {key}={cv_results[param_key][idx]}"
    
    summary += f"\n\n{'='*70}\n"
    
    # Salvar
    output_path = Path(output_dir) / f"{experiment_name}_summary.txt"
    with open(output_path, 'w') as f:
        f.write(summary)
    
    print(f"✓ Sumário salvo: {output_path}")
    print(summary)
    
    return output_path


if __name__ == "__main__":
    print("Módulo de persistência de CV results importado com sucesso.")
    print("Use: save_cv_results, plot_train_vs_validation_curves, plot_hyperparameter_impact")

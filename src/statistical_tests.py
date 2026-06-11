"""
Testes estatísticos pareados para comparação de modelos.
Implementa Wilcoxon pareado e paired t-test.
"""

import numpy as np
from scipy.stats import wilcoxon, ttest_rel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd


def paired_wilcoxon_test(baseline_scores, variant_scores):
    """
    Executa teste de Wilcoxon pareado para comparar two conditions.
    
    Parâmetros:
        baseline_scores: array de métricas do baseline (por fold)
        variant_scores: array de métricas da variante (por fold)
    
    Retorna:
        dict com statistic, p_value, conclusão
    """
    if len(baseline_scores) != len(variant_scores):
        raise ValueError("baseline_scores e variant_scores devem ter mesmo tamanho (folds pareados)")

    # Diferenças pareadas
    differences = np.array(variant_scores) - np.array(baseline_scores)

    if np.all(differences == 0):
        statistic, p_value = 0.0, 1.0  # idênticos, logo p-valor é 1 (sem diferença)
    else:
        statistic, p_value = wilcoxon(variant_scores, baseline_scores, alternative='greater')
    
    # Teste de Wilcoxon (não paramétrico, robusto a outliers)
    # 'greater' testa se a variante é significativamente maior que o baseline
    statistic, p_value = wilcoxon(variant_scores, baseline_scores, alternative='greater')
    
    # Interpretação
    mean_baseline = np.mean(baseline_scores)
    mean_variant = np.mean(variant_scores)
    std_baseline = np.std(baseline_scores, ddof=1)
    std_variant = np.std(variant_scores, ddof=1)
    
    improvement = mean_variant - mean_baseline
    significant = p_value < 0.05
    
    return {
        "test": "Wilcoxon Pareado",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": significant,
        "mean_baseline": float(mean_baseline),
        "std_baseline": float(std_baseline),
        "mean_variant": float(mean_variant),
        "std_variant": float(std_variant),
        "improvement": float(improvement),
        "improvement_pct": float((improvement / mean_baseline * 100) if mean_baseline != 0 else 0),
    }


def paired_ttest(baseline_scores, variant_scores):
    """
    Executa paired t-test (paramétrico).
    
    Parâmetros:
        baseline_scores: array de métricas do baseline (por fold)
        variant_scores: array de métricas da variante (por fold)
    
    Retorna:
        dict com t-statistic, p_value, conclusão
    """
    if len(baseline_scores) != len(variant_scores):
        raise ValueError("baseline_scores e variant_scores devem ter mesmo tamanho (folds pareados)")
    
    t_statistic, p_value = ttest_rel(baseline_scores, variant_scores)
    
    mean_baseline = np.mean(baseline_scores)
    mean_variant = np.mean(variant_scores)
    std_baseline = np.std(baseline_scores, ddof=1)
    std_variant = np.std(variant_scores, ddof=1)
    
    improvement = mean_variant - mean_baseline
    significant = p_value < 0.05
    
    return {
        "test": "Paired t-test",
        "t_statistic": float(t_statistic),
        "p_value": float(p_value),
        "significant": significant,
        "mean_baseline": float(mean_baseline),
        "std_baseline": float(std_baseline),
        "mean_variant": float(mean_variant),
        "std_variant": float(std_variant),
        "improvement": float(improvement),
        "improvement_pct": float((improvement / mean_baseline * 100) if mean_baseline != 0 else 0),
    }


def format_comparison_report(test_result, metric_name="F1-macro"):
    """
    Formata resultado do teste em texto legível.
    
    Exemplo:
        "F1-macro: baseline=0.812±0.014 vs variante=0.827±0.012 (melhoria=+1.85%)
         Wilcoxon p-value=0.038 (significativo)"
    """
    report = (
        f"{metric_name}:\n"
        f"  Baseline: {test_result['mean_baseline']:.4f} ± {test_result['std_baseline']:.4f}\n"
        f"  Variante: {test_result['mean_variant']:.4f} ± {test_result['std_variant']:.4f}\n"
        f"  Melhoria: {test_result['improvement']:+.4f} ({test_result['improvement_pct']:+.2f}%)\n"
        f"  Teste: {test_result['test']}\n"
        f"  p-value: {test_result['p_value']:.4f} "
        f"({'SIGNIFICATIVO' if test_result['significant'] else 'NÃO significativo'})\n"
    )
    return report


# Exemplo de uso
if __name__ == "__main__":
    # Simular 5 folds com scores pareados
    baseline_f1_scores = [0.810, 0.815, 0.805, 0.820, 0.812]  # 5 folds
    variant_f1_scores = [0.825, 0.830, 0.820, 0.835, 0.828]
    
    # Teste de Wilcoxon
    wilcoxon_result = paired_wilcoxon_test(baseline_f1_scores, variant_f1_scores)
    print("=" * 70)
    print("TESTE DE WILCOXON PAREADO")
    print("=" * 70)
    print(format_comparison_report(wilcoxon_result, "F1-macro"))
    
    # Teste t pareado
    ttest_result = paired_ttest(baseline_f1_scores, variant_f1_scores)
    print("=" * 70)
    print("PAIRED T-TEST")
    print("=" * 70)
    print(format_comparison_report(ttest_result, "F1-macro"))

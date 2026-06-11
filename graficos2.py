import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

variantes = [f'v{i}' for i in range(1, 13)]

dados_modelos = {
    'Variante': variantes,
    'Random Forest': [0.300, 0.401, 0.398, 0.421, 0.419, 0.389, 0.417, 0.425, 0.420, 0.305, 0.315, 0.319],
    'XGBoost':       [0.320, 0.398, 0.370, 0.389, 0.389, 0.372, 0.389, 0.389, 0.389, 0.303, 0.320, 0.323],
    'LightGBM':      [0.322, 0.387, 0.351, 0.369, 0.369, 0.352, 0.369, 0.369, 0.369, 0.274, 0.291, 0.296],
    'Voting':        [0.273, 0.295, 0.308, 0.324, 0.324, 0.309, 0.324, 0.324, 0.324, 0.253, 0.258, 0.272],
    'Decision Tree': [0.284, 0.344, 0.382, 0.401, 0.403, 0.381, 0.402, 0.402, 0.401, 0.310, 0.312, 0.315],
    'Stacking':      [0.297, 0.384, 0.293, 0.306, 0.307, 0.288, 0.304, 0.310, 0.305, 0.256, 0.260, 0.262],
    'KNN':           [0.213, 0.279, 0.292, 0.305, 0.305, 0.291, 0.305, 0.307, 0.305, 0.296, 0.302, 0.305],
    'AdaBoost':      [0.263, 0.258, 0.265, 0.274, 0.274, 0.265, 0.274, 0.274, 0.274, 0.188, 0.196, 0.200],
    'MLP':           [0.261, 0.293, 0.290, 0.327, 0.324, 0.303, 0.335, 0.325, 0.331, 0.304, 0.302, 0.310],
    'Logistic Regression': [0.194, 0.206, 0.203, 0.254, 0.253, 0.206, 0.254, 0.253, 0.254, 0.195, 0.202, 0.234],
    'Naive Bayes':   [0.157, 0.185, 0.200, 0.142, 0.142, 0.084, 0.142, 0.230, 0.142, 0.204, 0.212, 0.230],
    'LVQ':           [0.110, 0.150, 0.126, 0.114, 0.111, 0.135, 0.114, 0.175, 0.114, 0.114, 0.112, 0.107],
    'SVM':           [0.166, 0.078, 0.094, 0.083, 0.083, 0.090, 0.083, 0.119, 0.083, 0.100, 0.109, 0.119],
    'Bagging':       [0.313, 0.372, 0.405, 0.431, 0.430, 0.405, 0.430, 0.430, 0.428, 0.324, 0.327, 0.330]
}

df = pd.DataFrame(dados_modelos)

# Configuração do gráfico
plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid")

# Plotar a linha de cada modelo predileto
for coluna in df.columns[1:]:
    plt.plot(df['Variante'], df[coluna], marker='o', linewidth=2, label=coluna)

plt.axhline(y=0.398, color='black', linestyle='--', linewidth=1.5, label='Baseline Referência \n(Bagging - F1-macro = 0,398)')

plt.title('Evolução do F1-Macro: Impacto das Variantes (v1 a v12) vs. Baseline', fontsize=14, pad=15, weight='bold')
plt.xlabel('Variantes de Pré-processamento / Ablação', fontsize=12)
plt.ylabel('F1-Macro de Validação', fontsize=12)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11) # Legenda para fora do gráfico
plt.ylim(0.05, 0.50)

plt.text(1.25, 0.49, 'Fase 1', ha='center')
plt.text(5.5, 0.49, 'Fase 2', ha='center')
plt.text(10.0, 0.49, 'Fase 3', ha='center')

plt.axvspan(-0.5, 2.5, alpha=0.1)
plt.axvspan(2.5, 8.5, alpha=0.05)
plt.axvspan(8.5, 11.5, alpha=0.1)

plt.tight_layout()
plt.savefig('evolucao_variantes_vs_baseline.png', dpi=300)
plt.show()
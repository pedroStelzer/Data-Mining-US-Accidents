import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

dados = {
    'Modelo': ['Random Forest', 'Bagging', 'LightGBM', 'Voting', 'XGBoost', 
               'Decision Tree', 'Stacking', 'KNN', 'AdaBoost', 'MLP', 
               'Reg. Logística', 'Naive Bayes', 'LVQ', 'SVM'],
    'F1-Macro': [
        0.362,  # Random Forest
        0.398,  # Bagging
        0.356,  # LightGBM
        0.259,  # Voting
        0.374,  # XGBoost
        0.385,  # Decision Tree
        0.394,  # Stacking
        0.273,  # KNN
        0.245,  # AdaBoost
        0.267,  # MLP
        0.228,  # Reg. Logística (LogisticRegression)
        0.264,  # Naive Bayes
        0.110,  # LVQ
        0.238   # SVM
    ]
}

df = pd.DataFrame(dados)
df = df.sort_values(by='F1-Macro', ascending=False) # Garante a ordem do melhor para o pior

# 2. Configuração estética do gráfico
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

# Criando uma paleta de cores para destacar os 3 primeiros colocados
# Os 3 melhores ganham um azul escuro marcante, os outros ficam em cinza sutil
cores = ['#1f77b4' if i < 3 else '#b0bec5' for i in range(len(df))]

# Criando o gráfico de barras horizontais
ax = sns.barplot(x='F1-Macro', y='Modelo', data=df, palette=cores)

# Adicionando os valores numéricos na ponta de cada barra
for p in ax.patches:
    width = p.get_width()
    ax.text(width + 0.01, p.get_y() + p.get_height()/2 + 0.1, 
            f'{width:.3f}', ha="left", va="center", fontsize=10, weight='bold' if width >= 0.43 else 'normal')

# Títulos e labels
plt.title('Comparação de Desempenho dos Modelos Baseline (F1-Macro de Validação)', fontsize=14, pad=15, weight='bold')
plt.xlabel('F1-Macro de Validação', fontsize=12)
plt.ylabel('Modelos / Algoritmos', fontsize=12)
plt.xlim(0, 0.55)

plt.tight_layout()

# 3. Salva a imagem no formato ideal para o relatório
plt.savefig('f1_macro_baseline.png', dpi=300)
plt.show()
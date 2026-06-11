# Data-Mining-US-Accidents

## 📋 Descrição

Pipeline completo de **classificação de gravidade de acidentes de trânsito nos EUA** utilizando técnicas de Machine Learning e Data Mining. O projeto implementa:

- **13 algoritmos de classificação** (Random Forest, XGBoost, LightGBM, SVM, MLP, Voting, Stacking, etc.)
- **Busca automática de hiperparâmetros** (RandomizedSearchCV)
- **Engenharia de features** com múltiplas configurações de ablação
- **Balanceamento de dados** (oversample, undersample, SMOTE, severity sampling)
- **Rastreamento de experimentos** via MLflow
- **Testes estatísticos** (Wilcoxon pareado para validação)
- **Análise comparativa** entre modelos baseline e otimizados

### Objetivo Principal
Classificar acidentes em 4 níveis de gravidade (1-4) com base em características como tempo de duração, condições climáticas, infraestrutura e localização geográfica.

---

## 📁 Estrutura do Projeto

```
Data-Mining-US-Accidents/
├── main_pipeline.py                 # Orquestrador principal do pipeline
├── requirements.txt                 # Dependências do projeto
├── README.md                        # Este arquivo
│
├── src/                             # Código-fonte
│   ├── config.py                    # Carregamento de configurações
│   ├── config.yaml                  # Arquivo de configuração principal
│   ├── data_ingestion.py            # Carregamento e cache de dados
│   ├── cleaning.py                  # Limpeza de dados
│   ├── pipeline.py                  # Pipeline de preprocessamento
│   ├── transformers.py              # Transformadores customizados
│   ├── train.py                     # Treinamento de modelos (baseline e hyperparameter tuning)
│   ├── run_experiment.py            # Orquestração de experimentos
│   ├── evaluation.py                # Avaliação e comparação de modelos
│   ├── statistical_tests.py         # Testes estatísticos (Wilcoxon)
│   └── persistence.py               # Salvamento de resultados e gráficos
│
├── artifacts/                       # Modelos treinados salvos
│   ├── best_model/
│   ├── model_Bagging/
│   ├── model_LightGBM/
│   ├── model_RandomForest/
│   └── wilcoxon_report_*.txt
│
├── cv_results_logs/                 # Logs de validação cruzada
│   ├── *_cv_results.csv
│   └── *_summary.txt
│
├── final_evaluation/                # Avaliação final
│   ├── REPORT_*.txt
│   ├── confusion_matrix_*.png
│   ├── roc_curve_*.png
│   └── pr_curve_*.png
│
├── plots_hyperparameter/            # Visualizações de hiperparâmetros
├── mlruns/                          # MLflow tracking server
└── data/
    ├── raw/                         # Dados brutos (não versionados)
    └── processed/                   # Dados processados
```

---

## 🚀 Instalação

### Pré-requisitos
- Python 3.9+
- pip ou conda

### Passo 1: Clonar o repositório
```bash
git clone https://github.com/pedroStelzer/Data-Mining-US-Accidents.git
cd Data-Mining-US-Accidents
```

### Passo 2: Criar ambiente virtual
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### Passo 3: Instalar dependências
```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuração

Edite `src/config.yaml` para customizar:

```yaml
data_ingestion:
  sample_fraction: 0.05           # Fração de dados para treino (0.05 = 5%)

data_balancing:
  resampling_method: "severity_sampling"  # Método: none, oversample, undersample, smote
  
feature_engineering:
  duration: true                  # Engenharia de features de duração
  wind: true                      # Features de vento
  weather: true                   # Features climáticas
  geo: true                       # Features geográficas
  infrastructure: true            # Features de infraestrutura
  pca:
    enabled: false
    n_components: 0.90
```

---

## 🎯 Como Usar

### 1. Pipeline Completo (Padrão)
```bash
python main_pipeline.py
```

### 2. Com Busca de Hiperparâmetros
```bash
python main_pipeline.py \
  --experiment-group "hptuning_v1" \
  --resampling-method "oversample" \
  --use-pca \
  --pca-n-components 0.95
```

### 3. Apenas Baseline (sem tuning)
```bash
python main_pipeline.py \
  --experiment-group "baseline" \
  --resampling-method "none"
```

### 4. Ablação de Feature Engineering
```bash
python main_pipeline.py \
  --feature-engineering-ablation \
  --experiment-group "fe_ablation"
```

### Parâmetros Disponíveis

| Parâmetro | Opções | Descrição |
|-----------|--------|-----------|
| `--experiment-group` | string | Nome do grupo de experimentos (para tags MLflow) |
| `--resampling-method` | none, oversample, undersample, smote, severity_sampling | Método de balanceamento |
| `--use-pca` | flag | Ativa PCA como pré-processamento |
| `--pca-n-components` | 0.0-1.0 | Proporção da variância para PCA |
| `--feature-engineering-ablation` | flag | Executa estudo de ablação de FE |
| `--mlflow-tracking-uri` | sqlite:// ou file:// | URI para MLflow tracking |

---

## 🤖 Modelos Disponíveis

O pipeline treina os seguintes 13 algoritmos:

| Modelo | Tipo | n_iter | Status |
|--------|------|--------|--------|
| **RandomForest** | Ensemble | 15 | ✅ |
| **XGBoost** | Boosting | 12 | ✅ |
| **LightGBM** | Boosting | 12 | ✅ |
| **LogisticRegression** | Linear | 4 | ✅ |
| **DecisionTree** | Tree | 8 | ✅ |
| **KNN** | Distance-based | 2 | ✅ |
| **NaiveBayes** | Probabilistic | 3 | ✅ |
| **MLP** | Neural Network | 5 | ✅ |
| **SVM** | Kernel-based | 3 | ✅ |
| **AdaBoost** | Boosting | 8 | ✅ |
| **Bagging** | Ensemble | 8 | ✅ |
| **Voting** | Ensemble | 2 | ✅ |
| **Stacking** | Ensemble | 4 | ✅ |

---

## 📊 Rastreamento com MLflow

### Iniciar MLflow UI
```bash
mlflow ui
```
Acesse: `http://localhost:5000`

### Scripts de Extração de Dados

#### Extrair CV Scores e Wilcoxon P-value
```bash
python resultados.py
```
Gera CSVs para cada experimento com:
- Nome do modelo
- `best_cv_score_mean` (média dos 5 folds)
- `best_cv_score_std` (desvio padrão)
- Os 5 scores individuais dos folds
- `wilcoxon_p_value` (significância estatística)

#### Extrair com Hiperparâmetros
```bash
python resultados2.py
```
Gera CSVs com todos os hiperparâmetros otimizados para cada modelo.

---

## 📈 Resultados e Artefatos

### Após execução, os seguintes artefatos são gerados:

1. **Modelos Treinados** (`artifacts/`)
   - Modelos em pickle para cada algoritmo
   - Best model selecionado automaticamente

2. **Logs de Validação Cruzada** (`cv_results_logs/`)
   - CSV com scores de cada fold
   - Sumário em texto com estatísticas

3. **Avaliação Final** (`final_evaluation/`)
   - Relatório em texto (REPORT_*.txt)
   - Matriz de confusão (PNG)
   - Curva ROC (PNG)
   - Curva Precision-Recall (PNG)

4. **Rastreamento MLflow** (`mlruns/`)
   - Database SQLite com todos os experimentos
   - Métricas, parâmetros e artefatos salvos
   - Histórico completo de runs

### Exemplo de Relatório
```
MELHOR MODELO: LightGBM

MÉTRICAS NO CONJUNTO DE TESTE:
  Accuracy: 0.7823
  Balanced Accuracy: 0.6542
  F1-macro: 0.6231
  F1-weighted: 0.7654

RECOMENDAÇÕES:
  ✓ Modelo aprovado para deployment em produção
```

---

## 🧪 Testes Estatísticos

O pipeline realiza **Teste de Wilcoxon Pareado** para validar significância estatística:

```bash
# Automaticamente executado no pipeline
# Compara: Best Model Tunado vs Baseline Puro
# Resultado: p-value < 0.05 indica diferença significativa
```

Resultados salvos em:
- `artifacts/wilcoxon_report_*.txt`
- Armazenado também em MLflow com métrica `wilcoxon_p_value`

---

## 🔍 Exploração de Dados

### Estatísticas dos Dados

```python
from src.data_ingestion import load_data

X_train, X_val, X_test, y_train, y_val, y_test = load_data(sample_fraction=0.05)

print(f"Treino: {X_train.shape[0]} amostras")
print(f"Validação: {X_val.shape[0]} amostras")
print(f"Teste: {X_test.shape[0]} amostras")

# Distribuição de classes
print(y_train.value_counts())
```

---

## 📝 Dependências

Principais bibliotecas utilizadas:

```
scikit-learn>=1.5.2     # Machine Learning
xgboost                 # Boosting
lightgbm                # LightGBM
mlflow>=2.16.2          # Experiment Tracking
pandas>=2.2.2           # Manipulação de dados
numpy>=2.1.1            # Computação numérica
matplotlib>=3.9.2       # Visualização
seaborn>=0.13.2         # Visualização avançada
imbalanced-learn        # Resampling
sklvq                   # LVQ (opcional)
```

---

## 🐛 Troubleshooting

### Erro: "mlflow.db not found"
```bash
# Solução: Reset tracking URI
python -c "import mlflow; mlflow.set_tracking_uri('sqlite:///mlflow.db')"
```

### Erro: "XGBoost/LightGBM not found"
```bash
pip install xgboost lightgbm
```

---

## 📚 Referências

- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)

---

## 👤 Autores

**Pedro Stelzer** 
**Bruno Henrique** 
Universidade Federal de Pernambuco (UFPE)

---

## 📄 Licença

Este projeto está licenciado sob a [LICENSE](LICENSE) incluída no repositório.
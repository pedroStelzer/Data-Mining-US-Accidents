baseline = {
    "Random Forest": 0.362,
    "Bagging": 0.398,
    "LightGBM": 0.356,
    "Voting": 0.259,
    "XGBoost": 0.374,
    "Decision Tree": 0.385,
    "Stacking": 0.394,
    "KNN": 0.273,
    "AdaBoost": 0.245,
    "MLP": 0.267,
    "Logistic Regression": 0.228,
    "Naive Bayes": 0.264,
    "LVQ": 0.110,
    "SVM": 0.238
}

tuning = {
    "Random Forest": 0.4622,
    "Bagging": 0.4644,
    "LightGBM": 0.4560,
    "Voting": 0.4219,
    "XGBoost": 0.4010,
    "Decision Tree": 0.3809,
    "Stacking": 0.3402,
    "KNN": 0.3227,
    "AdaBoost": 0.2922,
    "MLP": 0.3165,
    "Logistic Regression": 0.2533,
    "Naive Bayes": 0.2304,
    "LVQ": 0.1772,
    "SVM": 0.1192
}

import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({
    "Modelo": baseline.keys(),
    "Baseline": baseline.values(),
    "Tuning": [tuning[m] for m in baseline.keys()]
})

df["Ganho"] = df["Tuning"] - df["Baseline"]
df = df.sort_values("Ganho")

plt.figure(figsize=(10,8))

for i, row in enumerate(df.itertuples()):
    plt.plot(
        [row.Baseline, row.Tuning],
        [i, i],
        linewidth=2
    )

plt.scatter(df["Baseline"], range(len(df)), s=80, label="Baseline")
plt.scatter(df["Tuning"], range(len(df)), s=80, label="Tuning")

plt.yticks(range(len(df)), df["Modelo"])
plt.xlabel("F1-Macro")
plt.title("Impacto do Hyperparameter Tuning")
plt.legend()

plt.tight_layout()
plt.savefig('grafico_ganho_hptuning.png', dpi=300)
plt.show()
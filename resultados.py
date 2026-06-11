import pandas as pd
import glob
import os

arquivos = glob.glob("results/*.csv")

dados = {}

for arq in arquivos:
    nome = os.path.splitext(os.path.basename(arq))[0]
    dados[nome] = pd.read_csv(arq)

baseline = dados["US_Accidents_Phase_balancing_method_none"]

resumo = []

for variante, df in dados.items():

    if variante == "baseline":
        continue

    merged = df.merge(
        baseline[["modelo", "best_cv_score_mean"]],
        on="modelo",
        suffixes=("_var", "_base")
    )

    merged["ganho"] = (
        merged["best_cv_score_mean_var"]
        - merged["best_cv_score_mean_base"]
    )

    n_melhoraram = (merged["ganho"] > 0).sum()

    n_significativos = (
        df["wilcoxon_p_value"] < 0.05
    ).sum()

    ganho_medio = merged["ganho"].mean()

    resumo.append({
        "variante": variante,
        "ganho_medio": ganho_medio,
        "n_melhoraram": n_melhoraram,
        "n_significativos": n_significativos
    })

resumo = pd.DataFrame(resumo)

melhor = resumo.loc[
    resumo["n_significativos"].idxmax()
]

melhor_variante = melhor["variante"]

df_melhor = dados[melhor_variante]

modelos_sig = df_melhor[
    df_melhor["wilcoxon_p_value"] < 0.05
]["modelo"].tolist()

texto = (
    f"Variante com maior ganho consistente: "
    f"{melhor_variante}. \n"
    f"Modelos que ganharam significância "
    f"(p < 0,05): {', '.join(modelos_sig)}.\n"
    f"{melhor}"
)

print(texto)
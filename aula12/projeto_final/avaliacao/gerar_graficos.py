"""Gera os graficos comparativos (barras por metrica + evolucao) a partir de
resultados.csv, em avaliacao/graficos/."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PASTA = Path(__file__).resolve().parent
PASTA_GRAFICOS = PASTA / "graficos"
PASTA_GRAFICOS.mkdir(exist_ok=True)

METRICAS = ["hit@5", "recall@5", "mrr", "ndcg@10"]
COR = "#4C72B0"


def carregar_linhas():
    with open(PASTA / "resultados.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def grafico_evolucao(linhas):
    linhas_validas = [l for l in linhas if to_float(l["ndcg@10"]) is not None]
    labels = [l["exp"] for l in linhas_validas]
    for metrica in METRICAS:
        valores = [to_float(l[metrica]) for l in linhas_validas]
        plt.figure(figsize=(max(10, len(labels) * 0.55), 5))
        plt.plot(labels, valores, marker="o", color=COR)
        plt.xticks(rotation=60, ha="right", fontsize=8)
        plt.ylabel(metrica)
        plt.title(f"Evolucao de {metrica} ao longo dos experimentos")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(PASTA_GRAFICOS / f"evolucao_{metrica.replace('@','_at_')}.png", dpi=130)
        plt.close()


def grafico_barras_fase(linhas, fase, nome_arquivo, rotulo_fn=None):
    subset = [l for l in linhas if l["fase"] == fase and to_float(l["ndcg@10"]) is not None]
    if not subset:
        return
    labels = [rotulo_fn(l) if rotulo_fn else l["exp"] for l in subset]
    fig, axes = plt.subplots(1, len(METRICAS), figsize=(4.2 * len(METRICAS), 4.5))
    for ax, metrica in zip(axes, METRICAS):
        valores = [to_float(l[metrica]) for l in subset]
        ax.bar(labels, valores, color=COR)
        ax.set_title(metrica)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")
    fig.suptitle(fase)
    plt.tight_layout()
    plt.savefig(PASTA_GRAFICOS / nome_arquivo, dpi=130)
    plt.close()


def grafico_ragas(linhas):
    subset = [l for l in linhas if l["fase"] == "Fase 8 - RAGAS" and l.get("ragas_faith")]
    if not subset:
        return
    labels = [l["exp"] for l in subset]
    cols = [("ragas_faith", "Faithfulness"), ("ragas_ans_rel", "Answer Relevancy"), ("ragas_ctx_recall", "Context Recall")]
    fig, axes = plt.subplots(1, len(cols), figsize=(4.5 * len(cols), 4.5))
    for ax, (col, nome) in zip(axes, cols):
        valores = [to_float(l[col]) for l in subset]
        ax.bar(labels, valores, color="#DD8452")
        ax.set_title(nome)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=20, labelsize=9)
    fig.suptitle("RAGAS - baseline vs melhor configuracao")
    plt.tight_layout()
    plt.savefig(PASTA_GRAFICOS / "ragas_comparativo.png", dpi=130)
    plt.close()


def main():
    linhas = carregar_linhas()
    grafico_evolucao(linhas)
    grafico_barras_fase(linhas, "Fase 2 - Chunking", "fase2_chunking.png",
                         rotulo_fn=lambda l: l["exp"].replace("exp_chunk_", ""))
    grafico_barras_fase(linhas, "Fase 5 - Query enhancement", "fase5_query_enhancement.png",
                         rotulo_fn=lambda l: l["exp"].replace("exp05_", ""))
    grafico_ragas(linhas)
    print(f"Graficos salvos em {PASTA_GRAFICOS}")


if __name__ == "__main__":
    main()

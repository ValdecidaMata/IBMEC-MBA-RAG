"""Helpers compartilhados pelos scripts de avaliacao/.

Reaproveita app.busca_avancada.construir(..., apenas_recuperacao=True) para rodar
qualquer tecnica (baseline/multi_query/rag_fusion/step_back/hibrida/rerank) sem
pagar o custo de uma geracao de resposta por pergunta. O matching pergunta->artigo
e feito por regex sobre o CONTEUDO do chunk recuperado (ver Contexto do plano:
`id_original` nunca e populado pelo projeto, entao nao da para usar como chave).
"""
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import ndcg_score

PASTA = Path(__file__).resolve().parent
RAIZ_PROJETO_FINAL = PASTA.parent
sys.path.insert(0, str(RAIZ_PROJETO_FINAL))

# Os scripts de avaliacao rodam retrieval puro fora do processo da API (uvicorn):
# desligamos o LangfuseConnector aqui (chaves da instancia local do docker-compose
# nao correspondem a nenhum projeto criado via UI) para nao falhar o pipeline por
# causa de uma dependencia so de observabilidade/bonus.
os.environ.setdefault("LANGFUSE_SECRET_KEY", "")
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "")

PADRAO_ARTIGO_CHUNK = re.compile(r"Art\.?\s*(\d+)[ºo°]?(-[A-Z])?", re.IGNORECASE)

COLUNAS_CSV = [
    "exp", "fase", "mudanca", "hit@5", "recall@5", "mrr", "ndcg@10",
    "ragas_faith", "ragas_ans_rel", "ragas_ctx_recall", "latencia_s", "custo", "observacao",
]


def carregar_dataset(caminho=None):
    caminho = caminho or (PASTA / "dataset.json")
    return json.loads(Path(caminho).read_text(encoding="utf-8"))


def artigos_no_chunk(texto: str) -> set:
    """IDs de artigo (ex.: 'Art5', 'Art149A') mencionados no inicio/corpo de um chunk."""
    ids = set()
    for m in PADRAO_ARTIGO_CHUNK.finditer(texto):
        sufixo = (m.group(2) or "").replace("-", "")
        ids.add(f"Art{m.group(1)}{sufixo}")
    return ids


_PIPE_CACHE = {}  # (tecnica, top_k) -> (pipe, chave) ; evita recarregar o reranker a cada pergunta


def _inputs_para(tecnica: str, pergunta: str) -> dict:
    """Recalcula so os `inputs` de construir() para uma nova pergunta, sem reconstruir
    o pipeline (o pipeline cacheado ja tem os componentes; so trocamos o texto)."""
    if tecnica == "baseline":
        return {"embedder": {"text": pergunta}}
    if tecnica == "hibrida":
        return {"hibrida": {"query": pergunta}}
    if tecnica == "rerank":
        return {"embedder": {"text": pergunta}, "ranker": {"query": pergunta}}
    return {"rw_prompt": {"pergunta": pergunta}, "montar": {"question": pergunta}}  # multi_query/rag_fusion/step_back


def recuperar(tecnica: str, top_k: int, pergunta: str):
    """Roda uma tecnica de busca (retrieval-only) e devolve a lista de haystack.Document."""
    from app import busca_avancada as ba

    chave_cache = (tecnica, top_k)
    if chave_cache in _PIPE_CACHE:
        pipe, chave = _PIPE_CACHE[chave_cache]
        inputs = _inputs_para(tecnica, pergunta)
    else:
        pipe, inputs, chave = ba.construir(tecnica, top_k, pergunta, apenas_recuperacao=True)
        _PIPE_CACHE[chave_cache] = (pipe, chave)
    saida = pipe.run(inputs, include_outputs_from={chave})
    return saida[chave]["documents"]


def metricas_recuperacao(ranked_ids_por_query: list[list[set]], golds: list[dict], k5=5, k10=10):
    """
    ranked_ids_por_query: por query, lista ORDENADA de sets de artigo-ids (1 set por
    chunk recuperado, na ordem de ranking).
    golds: por query, dict {artigo_id: nota} (nota>0 = relevante).
    Calcula Hit@5, Recall@5, MRR (sobre toda a lista) e NDCG@10 (graduado).
    """
    hits5, recalls5, rrs = [], [], []
    y_true_rows, y_score_rows = [], []

    for chunk_ids_ranked, gold in zip(ranked_ids_por_query, golds):
        relevantes = {a for a, nota in gold.items() if nota > 0}
        if not relevantes:
            continue

        cobertos_top5 = set()
        for chunk_ids in chunk_ids_ranked[:k5]:
            cobertos_top5 |= chunk_ids
        acertos5 = cobertos_top5 & relevantes
        hits5.append(1.0 if acertos5 else 0.0)
        recalls5.append(len(acertos5) / len(relevantes))

        rr = 0.0
        for pos, chunk_ids in enumerate(chunk_ids_ranked, start=1):
            if chunk_ids & relevantes:
                rr = 1.0 / pos
                break
        rrs.append(rr)

        # NDCG@10: para cada chunk no top-10, nota = maior grau de relevancia dentre
        # os artigos que ele cobre (0 se nenhum for relevante/gold).
        y_true, y_score = [], []
        n = max(len(chunk_ids_ranked[:k10]), 1)
        for pos, chunk_ids in enumerate(chunk_ids_ranked[:k10]):
            grau = max((gold.get(a, 0) for a in chunk_ids), default=0)
            y_true.append(grau)
            y_score.append(n - pos)  # score decrescente = posicao do ranking
        if len(y_true) < 2:
            y_true += [0] * (2 - len(y_true))
            y_score += [0] * (2 - len(y_score))
        y_true_rows.append(y_true)
        y_score_rows.append(y_score)

    max_len = max(len(r) for r in y_true_rows) if y_true_rows else 0
    y_true_mat = np.array([r + [0] * (max_len - len(r)) for r in y_true_rows])
    y_score_mat = np.array([r + [0] * (max_len - len(r)) for r in y_score_rows])
    ndcg = ndcg_score(y_true_mat, y_score_mat, k=k10) if len(y_true_rows) else 0.0

    return {
        "hit@5": round(float(np.mean(hits5)), 4) if hits5 else 0.0,
        "recall@5": round(float(np.mean(recalls5)), 4) if recalls5 else 0.0,
        "mrr": round(float(np.mean(rrs)), 4) if rrs else 0.0,
        "ndcg@10": round(float(ndcg), 4),
        "n_queries": len(hits5),
    }


def avaliar_tecnica(tecnica: str, top_k: int, dataset: dict, k5=5, k10=10, top_k_busca=None):
    """Roda `tecnica` para todas as queries do dataset e calcula as metricas de recuperacao."""
    top_k_busca = top_k_busca or max(top_k, k10)
    ranked_ids_por_query, golds = [], []
    t0 = time.time()
    for q in dataset["queries_benchmark"]:
        docs = recuperar(tecnica, top_k_busca, q["query"])
        chunk_ids_ranked = [artigos_no_chunk(d.content) for d in docs]
        ranked_ids_por_query.append(chunk_ids_ranked)
        golds.append(q["relevancia"] if isinstance(q["relevancia"], dict) else {a: 1 for a in q["relevancia"]})
    dt = time.time() - t0
    m = metricas_recuperacao(ranked_ids_por_query, golds, k5=k5, k10=k10)
    m["latencia_s"] = round(dt / max(len(dataset["queries_benchmark"]), 1), 2)
    return m


def gravar_linha_csv(caminho_csv, linha: dict):
    caminho_csv = Path(caminho_csv)
    novo = not caminho_csv.exists()
    with open(caminho_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUNAS_CSV)
        if novo:
            w.writeheader()
        w.writerow({c: linha.get(c, "") for c in COLUNAS_CSV})

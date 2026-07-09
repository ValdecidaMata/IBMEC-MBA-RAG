"""Fase 8 - RAGAS: Faithfulness, ResponseRelevancy(strictness=1), LLMContextRecall,
LLMContextPrecisionWithReference. Roda o pipeline COMPLETO (busca + geracao, via
app.busca_avancada.construir sem apenas_recuperacao) e avalia com Groq como juiz +
Ollama para embeddings (padrao das Aulas 5/8).

Uso:
    python avaliar_ragas.py <exp_id> "<mudanca>" <tecnica> <top_k>
"""
import json
import os
import sys
import time
from pathlib import Path

PASTA = Path(__file__).resolve().parent
RAIZ_PROJETO_FINAL = PASTA.parent
sys.path.insert(0, str(RAIZ_PROJETO_FINAL))
os.environ.setdefault("LANGFUSE_SECRET_KEY", "")
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "")

import comum  # noqa: E402


_PIPE_CACHE = {}  # (tecnica, top_k) -> (pipe, chave) ; evita recarregar o reranker a cada pergunta


def responder_completo(tecnica: str, top_k: int, pergunta: str):
    from app import busca_avancada as ba

    chave_cache = (tecnica, top_k)
    if chave_cache in _PIPE_CACHE:
        pipe, chave = _PIPE_CACHE[chave_cache]
        inputs = comum._inputs_para(tecnica, pergunta)
        inputs["prompt"] = {"pergunta": pergunta}
    else:
        pipe, inputs, chave = ba.construir(tecnica, top_k, pergunta, apenas_recuperacao=False)
        _PIPE_CACHE[chave_cache] = (pipe, chave)
    saida = pipe.run(inputs, include_outputs_from={chave, "llm"})
    docs = saida[chave]["documents"]
    resposta = saida["llm"]["replies"][0]
    return resposta, [d.content for d in docs]


def coletar_amostras(dataset, tecnica, top_k, limite=0):
    queries = dataset["queries_benchmark"]
    if limite:
        queries = queries[:limite]
    amostras = []
    for i, q in enumerate(queries, 1):
        t0 = time.time()
        resposta, contextos = responder_completo(tecnica, top_k, q["query"])
        print(f"[{i}/{len(queries)}] ({time.time()-t0:.1f}s) {q['query'][:70]}")
        amostras.append({
            "question": q["query"], "contexts": contextos or ["(sem contexto)"],
            "answer": resposta, "ground_truth": q["resposta_referencia"],
        })
    return amostras


def rodar_ragas(amostras):
    from langchain_groq import ChatGroq
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy

    from app import config

    api_key, modelo, base_url = config.config_llm()
    juiz = LangchainLLMWrapper(ChatGroq(model=modelo, api_key=api_key, temperature=0))

    base_ollama, modelo_emb = config.config_ollama()
    try:
        from langchain_ollama import OllamaEmbeddings
    except ImportError:
        from langchain_community.embeddings import OllamaEmbeddings
    emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=modelo_emb, base_url=base_ollama))

    samples = [
        SingleTurnSample(user_input=a["question"], retrieved_contexts=a["contexts"],
                          response=a["answer"], reference=a["ground_truth"])
        for a in amostras
    ]
    dataset_ragas = EvaluationDataset(samples=samples)
    # strictness=1: Groq so aceita n=1 (o padrao do RAGAS pede n=3 e quebra o job).
    metricas = [Faithfulness(), ResponseRelevancy(strictness=1), LLMContextRecall(),
                LLMContextPrecisionWithReference()]
    resultado = evaluate(dataset=dataset_ragas, metrics=metricas, llm=juiz, embeddings=emb)
    return resultado.to_pandas()


def main():
    exp, mudanca, tecnica, top_k = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    limite = int(sys.argv[5]) if len(sys.argv) > 5 else 0

    dataset = comum.carregar_dataset()
    print(f"Coletando respostas (tecnica={tecnica}, top_k={top_k})...")
    amostras = coletar_amostras(dataset, tecnica, top_k, limite=limite)

    print("Rodando RAGAS (juiz Groq + embeddings Ollama)...")
    df = rodar_ragas(amostras)
    df.to_csv(PASTA / f"ragas_{exp}.csv", index=False, encoding="utf-8")

    def media(col):
        return float(df[col].mean()) if col in df.columns else float("nan")

    medias = {
        "faithfulness": media("faithfulness"),
        "answer_relevancy": media("answer_relevancy"),
        "context_recall": media("context_recall"),
        "llm_context_precision_with_reference": media("llm_context_precision_with_reference"),
        "n_amostras": len(amostras),
    }
    (PASTA / f"ragas_{exp}_medias.json").write_text(json.dumps(medias, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(medias, ensure_ascii=False, indent=2))

    linha = {
        "exp": exp, "fase": "Fase 8 - RAGAS", "mudanca": mudanca,
        "ragas_faith": round(medias["faithfulness"], 4),
        "ragas_ans_rel": round(medias["answer_relevancy"], 4),
        "ragas_ctx_recall": round(medias["context_recall"], 4),
        "observacao": f"ctx_precision_ref={medias['llm_context_precision_with_reference']:.4f}",
    }
    comum.gravar_linha_csv("resultados.csv", linha)


if __name__ == "__main__":
    main()

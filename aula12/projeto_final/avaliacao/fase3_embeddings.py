"""Fase 3 — compara modelos de embedding (Ollama) no corpus de artigos da
Constituicao, reaproveitando bench_embeddings/app (sem modifica-lo).
"""
import json
import sys
from pathlib import Path

PASTA = Path(__file__).resolve().parent
RAIZ = PASTA.parent.parent.parent  # .../IBMEC-MBA-RAG
sys.path.insert(0, str(RAIZ / "bench_embeddings"))

from app import dados, avaliacao  # bench_embeddings/app

MODELOS = ["nomic-embed-text", "bge-m3", "mxbai-embed-large"]
K = 10


def main():
    ds = json.loads((PASTA / "dataset.json").read_text(encoding="utf-8"))
    corpus_ids, corpus_texts, queries = dados._normalizar(ds)
    print(f"corpus: {len(corpus_ids)} artigos | queries: {len(queries)}")

    resultado = avaliacao.avaliar(corpus_ids, corpus_texts, queries, MODELOS, k=K)
    (PASTA / "fase3_resultado_embeddings.json").write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(resultado["resultados"], ensure_ascii=False, indent=2))
    print("\nMELHOR:", resultado["melhor"])
    print("\n", resultado["explicacao"])


if __name__ == "__main__":
    main()

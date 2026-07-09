"""Reindexa a Constituicao no OpenSearch com uma config controlada (chunking,
embedding, extracao), limpando o indice antes (Fase 2/3 pedem controle experimental:
"esquecer de reindexar" e uma das armadilhas do roteiro).

Uso:
    python reindexar.py --chunking hierarquico --embedding nomic-embed-text --fonte pymupdf
"""
import argparse
import json
import os
import sys
from pathlib import Path

import requests

PASTA = Path(__file__).resolve().parent
RAIZ_PROJETO_FINAL = PASTA.parent
sys.path.insert(0, str(RAIZ_PROJETO_FINAL))

os.environ.setdefault("LANGFUSE_SECRET_KEY", "")
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "")

CAMINHO_PDF = RAIZ_PROJETO_FINAL / "uploads" / "Constituicao-Compilado.pdf"


def texto_pymupdf(caminho_pdf) -> str:
    import fitz

    doc = fitz.open(str(caminho_pdf))
    return "".join(p.get_text() for p in doc)


def texto_docling(caminho_pdf) -> str:
    from app import extracao

    _, _, _, _, dados = extracao.decidir_e_extrair(str(caminho_pdf))
    return dados["conteudo"]


def limpar_indice():
    from app import config

    indice = config.config_opensearch()["indice"]
    url = f"{config.config_opensearch()['url']}/{indice}"
    r = requests.delete(url)
    print(f"DELETE {url} -> {r.status_code}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunking", default="auto",
                     choices=["auto", "fixo", "recursivo", "sentenca_janela", "semantico", "hierarquico"])
    ap.add_argument("--embedding", default=None, help="sobrescreve EMBEDDING_MODEL (ex.: bge-m3)")
    ap.add_argument("--fonte", default="pymupdf", choices=["pymupdf", "docling"])
    ap.add_argument("--sem-limpar", action="store_true", help="nao limpa o indice antes")
    args = ap.parse_args()

    if args.embedding:
        os.environ["EMBEDDING_MODEL"] = args.embedding

    from app import indexacao  # importado apos setar EMBEDDING_MODEL

    if not args.sem_limpar:
        limpar_indice()

    conteudo = texto_docling(CAMINHO_PDF) if args.fonte == "docling" else texto_pymupdf(CAMINHO_PDF)
    dados = {"conteudo": conteudo, "tabelas": [], "tecnica": args.fonte}
    meta = {"arquivo": "Constituicao-Compilado.pdf"}

    resultado = indexacao.indexar(dados, meta, destino_override="opensearch", chunking_override=args.chunking)
    resultado["n_chars_extraidos"] = len(conteudo)
    resultado["embedding"] = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
    print(json.dumps(resultado, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

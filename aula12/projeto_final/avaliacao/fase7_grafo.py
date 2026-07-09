"""Fase 7 - Avancada: ingere um subconjunto coeso (Titulo II - Art. 5 a 17, direitos e
garantias fundamentais) no LightRAG (destino=grafo) e compara respostas com o
OpenSearch nas perguntas multi-hop do dataset.

Escopo reduzido por custo/tempo: indexar a Constituicao INTEIRA no LightRAG geraria
centenas de chamadas de LLM (extracao de entidades/relacoes) via Groq. O Titulo II
e autocontido e rico em remissoes internas (bom para multi-hop) - ver plano.
"""
import json
import os
import sys
from pathlib import Path

PASTA = Path(__file__).resolve().parent
RAIZ_PROJETO_FINAL = PASTA.parent
sys.path.insert(0, str(RAIZ_PROJETO_FINAL))
os.environ.setdefault("LANGFUSE_SECRET_KEY", "")
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "")


def ingerir():
    """Ingestao com concorrencia reduzida (max_async=1): o tier gratuito da Groq
    (12000 TPM) estourou com a concorrencia padrao do LightRAG (max_async=4),
    derrubando a extracao de entidades por rate limit (429) - ver fase7_ingest.log
    da 1a tentativa. Aqui construimos o LightRAG manualmente (mesmo padrao de
    app/indexacao.py::_criar_lightrag) so para poder baixar a concorrencia."""
    import asyncio
    from functools import partial

    from app import config, indexacao

    async def _run():
        from lightrag import LightRAG
        from lightrag.llm.ollama import ollama_embed
        from lightrag.llm.openai import openai_complete_if_cache
        from lightrag.utils import EmbeddingFunc

        api_key, modelo, base_url = config.config_llm()
        o_base, o_modelo = config.config_ollama()

        async def llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
            return await openai_complete_if_cache(modelo, prompt, system_prompt=system_prompt,
                                                  history_messages=history_messages or [],
                                                  api_key=api_key, base_url=base_url, **kwargs)

        rag = LightRAG(working_dir=str(config.PASTA_RAG_STORAGE), llm_model_func=llm_func,
                       llm_model_max_async=1,  # 1 chamada por vez -> evita estourar TPM da Groq
                       embedding_func=EmbeddingFunc(embedding_dim=config.dimensao_embedding(),
                           max_token_size=8192,
                           func=partial(ollama_embed.func, embed_model=o_modelo, host=o_base)))
        await rag.initialize_storages()
        try:
            texto = (PASTA / "titulo2_texto.txt").read_text(encoding="utf-8")
            await rag.ainsert(texto)
        finally:
            await rag.finalize_storages()

    indexacao.rodar_async(_run)
    print("Ingestao concluida (max_async=1).")


def perguntar(pergunta, destino):
    from app import consulta

    # o indice OpenSearch (melhor config) usa bge-m3 (1024d); o grafo foi construido
    # com o embedding default (nomic-embed-text, 768d) - cada destino usa o seu.
    os.environ["EMBEDDING_MODEL"] = "bge-m3" if destino == "opensearch" else "nomic-embed-text"
    resposta, fontes, destino_usado = consulta.consultar(pergunta, destino=destino, top_k=5, tecnica="baseline")
    return resposta, fontes, destino_usado


def perguntar_grafo_subprocesso(pergunta):
    """Roda 1 pergunta no grafo num processo Python isolado: chamar consultar_grafo
    varias vezes no MESMO processo quebra com 'Lock ... bound to a different event
    loop' (LightRAG guarda locks globais presos ao primeiro asyncio.run()). Isolar
    por processo evita o problema sem mexer no LightRAG."""
    import subprocess

    codigo = (
        "import sys, os, json; sys.path.insert(0, r'" + str(RAIZ_PROJETO_FINAL) + "'); "
        "os.environ.setdefault('LANGFUSE_SECRET_KEY',''); os.environ.setdefault('LANGFUSE_PUBLIC_KEY',''); "
        "os.environ['EMBEDDING_MODEL']='nomic-embed-text'; "
        "from app import consulta; "
        "r, f, d = consulta.consultar(json.loads(sys.argv[1]), destino='grafo', top_k=5, tecnica='baseline'); "
        "print('###RESP###' + r)"
    )
    r = subprocess.run(
        [sys.executable, "-c", codigo, json.dumps(pergunta)],
        capture_output=True, text=True, cwd=str(RAIZ_PROJETO_FINAL), timeout=180,
    )
    saida = r.stdout
    if "###RESP###" in saida:
        return saida.split("###RESP###", 1)[1].strip()
    return f"(falhou: {r.stderr[-500:] if r.stderr else 'sem saida'})"


def comparar():
    dataset = json.loads((PASTA / "dataset.json").read_text(encoding="utf-8"))
    multi_hop = [q for q in dataset["queries_benchmark"] if q["tipo"] == "multi_hop"]
    # so as que fazem sentido dentro do Titulo II (Art. 5-17): Q21 (direitos+crianca fora do
    # titulo, mas Art.5 esta dentro) - usamos as que tem overlap com Art.5-17.
    alvo_ids = {f"Art{n}" for n in range(5, 18)}
    candidatas = [q for q in multi_hop if set(q["relevancia"]) & alvo_ids]

    resultado = []
    for q in candidatas:
        r_open, f_open, _ = perguntar(q["query"], "opensearch")
        r_graf = perguntar_grafo_subprocesso(q["query"])
        resultado.append({
            "id": q["id"], "query": q["query"], "resposta_referencia": q["resposta_referencia"],
            "opensearch": r_open, "grafo": r_graf,
        })
        print(f"\n=== {q['id']}: {q['query']} ===")
        print("REF :", q["resposta_referencia"])
        print("OS  :", r_open[:300])
        print("GRAF:", r_graf[:300])

    (PASTA / "fase7_comparacao_grafo.json").write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "comparar":
        comparar()
    else:
        ingerir()

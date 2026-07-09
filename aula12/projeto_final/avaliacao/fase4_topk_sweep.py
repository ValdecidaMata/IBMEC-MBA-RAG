import comum

dataset = comum.carregar_dataset()
for k in (3, 5, 10):
    m = comum.avaliar_tecnica("baseline", k, dataset, k5=k, k10=k, top_k_busca=k)
    linha = {
        "exp": f"exp04_topk{k}_real", "fase": "Fase 4 - top_k (retrieval real)",
        "mudanca": f"top_k={k} (retrieval real, Hit@{k}/Recall@{k}/NDCG@{k}), chunking=fixo, embedding=bge-m3",
        f"hit@5": m.get(f"hit@5", m.get("hit@5")), "recall@5": m["recall@5"], "mrr": m["mrr"], "ndcg@10": m["ndcg@10"],
        "latencia_s": m["latencia_s"], "observacao": f"metricas calculadas em @{k} (nao @5/@10 fixos)",
    }
    print(f"k={k}:", m)
    comum.gravar_linha_csv("resultados.csv", linha)

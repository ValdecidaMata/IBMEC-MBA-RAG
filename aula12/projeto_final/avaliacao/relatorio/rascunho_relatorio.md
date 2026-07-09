# Rascunho — Jornada de Melhoria da Recuperação em RAG (Constituição Federal)

## 1. Capa
- Fonte escolhida: Constituição da República Federativa do Brasil de 1988 (texto compilado, `Constituicao-Compilado.pdf`, 164 páginas, 250 artigos no corpo principal).
- Por quê: texto jurídico denso, estruturado em artigos/incisos, com bastante remissão interna — bom terreno para comparar chunking, embeddings e técnicas de recuperação, e para testar grafo (LightRAG) em perguntas multi-hop.

## 2. A fonte e o dataset de avaliação
- Corpus de gabarito: 274 "documentos" = 1 por artigo (250 números de artigo únicos, incluindo sufixos de emenda como Art. 149-A a 149-G), extraídos via regex `Art\.?\s*(\d+)[º°]?(-[A-Z])?` sobre o texto completo (exclui o ADCT, que reinicia a numeração de artigos, evitando colisão de IDs).
- 22 perguntas em `avaliacao/dataset.json`: 9 factuais (1 artigo), 6 reformuláveis (vocabulário coloquial, sem copiar termos do texto), 7 multi-hop/temáticas (2-3 artigos).
- Perguntas escritas manualmente e verificadas contra o conteúdo real dos artigos extraídos (evita o viés "pergunta copia o texto" apontado no roteiro).
- Matching pergunta→artigo: como `id_original` nunca é populado pelo projeto original (`consulta.py` lê o campo mas nada o define — chunk só tem `meta={"arquivo": ...}`), o script de avaliação identifica os artigos cobertos por um chunk recuperado via regex sobre o próprio conteúdo do chunk. Funciona igual para qualquer técnica de chunking, sem exigir mudança no pipeline de indexação.

## 3. Metodologia
- Métricas de recuperação: Hit@5, Recall@5, MRR (sobre toda a lista), NDCG@10 (graduado: nota 2 = muito relevante, nota 1 = relevante) — implementadas em `avaliacao/comum.py` (NDCG via `sklearn.metrics.ndcg_score`).
- Avaliação roda **sem** o nó de geração do RAG (parâmetro novo `apenas_recuperacao=True` em `busca_avancada.construir`), evitando gastar uma chamada de LLM por pergunta só para medir recuperação — as técnicas de reescrita de consulta (multi_query/rag_fusion/step_back) continuam fazendo sua própria chamada de LLM, pois isso é parte da técnica.
- Cada experimento reindexa o OpenSearch do zero (`avaliacao/reindexar.py` limpa o índice antes) para garantir comparação controlada.
- Ferramentas reaproveitadas sem modificação: `bench_embeddings/app` (comparação de modelos de embedding), padrões de `aula4` (híbrida BM25+RRF) e `aula3` (reranking) para as novas técnicas, padrão RAGAS+Groq de `aula5`/`aula8`.
- LangFuse: instância local subida via `docker-compose` (aula12/datasets), mas as chaves do `.env` do projeto final não correspondiam a um projeto criado nessa instância nova — tracing ficou fora do escopo desta rodada (ver limitações).

## 4. Baseline (Fase 0)
Rodado "como veio": chunking=auto, estratégia=opensearch, embedding=nomic-embed-text, técnica=baseline, top_k=5.

Resultado: Hit@5=0.1818, Recall@5=0.1364, MRR=0.0785, NDCG@10=0.1148.

**Achado central:** o agente de extração escolheu `extrair_texto` (Docling), que **falhou silenciosamente** a partir da página 82/164 (`std::bad_alloc` no modelo de layout, repetido para as páginas 82-164) — sem lançar exceção para o restante do pipeline, então o fallback para PyMuPDF (`extracao.py`, threshold de <50 caracteres) nunca foi acionado. Resultado: só 127-158 mil de 698 mil caracteres extraídos (~18-23%), faltando ~88% dos números de artigo no índice.

## 5. Experimentos

### Fase 1 — Extração
Hipótese: texto mal extraído limita tudo a jusante.
Mudança: `avaliacao/reindexar.py --fonte pymupdf` (substitui a extração por PyMuPDF completo, 698 372 caracteres, 100% dos 250 artigos).

| Config | Hit@5 | Recall@5 | MRR | NDCG@10 |
|---|---|---|---|---|
| exp01 baseline (Docling truncado) | 0.1818 | 0.1364 | 0.0785 | 0.1148 |
| exp02b PyMuPDF + chunking igual (hierárquico) — **isolado** | 0.3182 | 0.2879 | 0.2261 | 0.2984 |
| exp02 PyMuPDF + chunking=auto (sentença_janela) | 0.4091 | 0.3485 | 0.2456 | 0.3133 |

Análise: isolando só a extração (exp02b vs exp01), todas as métricas praticamente dobram/triplicam. Achado didático mais forte da jornada: um "bug" de extração (não uma questão de qualidade fina) derrubou o Recall em ~80%, e o próprio pipeline do projeto não teve como perceber sozinho (a falha por página é silenciosa e o limiar de fallback do `extracao.py` não cobre truncamento parcial).

### Fase 2 — Chunking
Mudança: `chunking` forçado nas 5 técnicas (`app/indexacao.py::chunkar`), extração e embedding fixos (PyMuPDF, nomic-embed-text).

| Técnica | n_chunks | Hit@5 | Recall@5 | MRR | NDCG@10 |
|---|---|---|---|---|---|
| fixo | 509 | 0.5 | 0.447 | 0.3415 | **0.4166** |
| recursivo | 686 | 0.5 | 0.3939 | 0.3438 | 0.403 |
| sentença_janela | 1131 | 0.4091 | 0.3485 | 0.2456 | 0.3133 |
| hierárquico | 1018 | 0.3182 | 0.2879 | 0.2261 | 0.2984 |
| semântico | 514 | 0.3182 | 0.2348 | 0.2087 | 0.256 |

Análise: contra a intuição de que técnicas "mais sofisticadas" (semântica, hierárquica) ganhariam, quem venceu foi o chunking **fixo** (200 palavras). Hipótese: artigos da Constituição variam muito de tamanho (Art. 5º sozinho tem 78 incisos e ~14 mil caracteres); janelas fixas de 200 palavras cortam o texto em unidades mais homogêneas e "densas" por chunk, enquanto sentença_janela fragmenta demais um artigo longo (muitos chunks quase-duplicados) e hierárquico/semântico às vezes agrupam vários artigos curtos, diluindo a especificidade do embedding.

### Fase 3 — Embedding
Bancada (`bench_embeddings`, 274 artigos x 22 perguntas, NDCG@10):

| Modelo | Hit@10 | Recall@10 | MRR | NDCG@10 | AUC |
|---|---|---|---|---|---|
| nomic-embed-text (768d) | 0.8182 | 0.7121 | 0.5278 | 0.5308 | 0.9292 |
| **bge-m3 (1024d)** | **1.0** | **0.9848** | **0.9356** | **0.9215** | **0.993** |
| mxbai-embed-large (1024d) | 0.8182 | 0.7197 | 0.5829 | 0.5433 | 0.9589 |

bge-m3 vence disparado. Reindexando o app (chunking=fixo + bge-m3): Hit@5=0.8182, Recall@5=0.7273, MRR=0.6458, NDCG@10=0.7045 — quase o dobro do melhor resultado da Fase 2. mxbai-embed-large tem contexto de 512 tokens (bem menor que bge-m3/nomic, 8192) — pode truncar artigos longos, o que ajuda a explicar seu desempenho fraco.

### Fase 4 — top_k e busca híbrida
top_k real (Hit@k/Recall@k/NDCG@k no próprio k, chunking=fixo+bge-m3):

| top_k | Hit@k | Recall@k | MRR | NDCG@k |
|---|---|---|---|---|
| 3 | 0.7273 | 0.6439 | 0.6061 | 0.6374 |
| 5 | 0.8182 | 0.7273 | 0.6288 | 0.6663 |
| 10 | 0.9545 | 0.8864 | 0.6458 | 0.7045 |

Recall sobe consistentemente com top_k (mais chunks candidatos = mais chance de cobrir o artigo certo), com custo de mais contexto (latência/tokens) na geração.

Nova técnica `hibrida` (BM25 + denso, RRF — `OpenSearchHybridRetriever`, `app/busca_avancada.py`), top_k=5: Hit@5=0.8636, Recall@5=0.7348, MRR=0.6788, NDCG@10=0.7089 — melhora sobre a busca densa pura (bge-m3 sozinho), a melhor config até aqui.

### Fase 5 — Query enhancement
top_k=5, chunking=fixo, embedding=bge-m3:

| Técnica | Hit@5 | Recall@5 | MRR | NDCG@10 |
|---|---|---|---|---|
| baseline (densa) | 0.8182 | 0.7273 | 0.6458 | 0.7045 |
| **hibrida** | **0.8636** | **0.7348** | **0.6788** | **0.7089** |
| multi_query | 0.8182 | 0.7121 | 0.5996 | 0.6687 |
| rag_fusion | 0.7727 | 0.6515 | 0.5522 | 0.6408 |
| step_back | 0.8182 | 0.7273 | 0.6201 | 0.6963 |

Análise: nenhuma técnica de reescrita de consulta supera a busca híbrida (ou mesmo a densa pura). Quando o embedding de base já é forte (bge-m3) e as perguntas do gabarito, embora coloquiais, são objetivas, reescrever a consulta introduz variações que dispersam o ranking (MRR cai em todas). O ganho de multi-query/RAG-fusion tende a aparecer quando o embedding de base é fraco ou a pergunta é muito ambígua — não é o caso aqui.

## (continua: Fase 6 Reranking, Fase 7 Grafo, Fase 8 RAGAS, tabela consolidada, gráficos, melhor config final, análise crítica, conclusão)

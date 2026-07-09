# Plano de Estudos Detalhado — MBA em RAG & CAG Aplicados a Direito e Segurança Pública

> Documento de estudo gerado a partir do material das 12 aulas + material complementar de Docling.
> Objetivo: entender **profundamente o mecanismo** de cada técnica (não só o "o que é"), para que você
> consiga reimplementar, adaptar e combinar essas técnicas em soluções novas, fora do domínio jurídico se necessário.

---

## Como usar este plano

Para cada aula você tem quatro camadas de estudo:

1. **Conceito e motivação** — que problema concreto a técnica resolve, e por que as técnicas anteriores falham.
2. **Mecanismo interno** — o algoritmo/matemática por trás, com fórmulas e pseudocódigo quando relevante.
3. **Onde praticar no repositório** — teoria (`teoria/AULAx_TEORIA.md`), exemplos e labs (notebooks), scripts (`.py`) reutilizáveis.
4. **Como reaproveitar em soluções novas** — o "gancho de engenharia": o que copiar/adaptar quando você for construir seu próprio sistema RAG.

Regra de ouro do curso, que se confirma aula após aula: **a qualidade de um RAG é determinada mais pela qualidade do retrieval (chunking, indexação, busca, reranking) do que pela escolha do LLM gerador.** Guarde essa frase — ela explica por que 9 das 12 aulas giram em torno de retrieval, e só depois disso entram agentes e orquestração.

### Cronograma sugerido (auto-ritmo, ~60h totais)

| Semana | Aulas | Foco | Horas |
|---|---|---|---|
| 1 | Aula 1 + Aula 2 | Fundamentos NLP/Transformers/LLM + Chunking/Docling/Naive RAG | 10h |
| 2 | Aula 3 + Aula 4 | Advanced RAG (rewriting, reranking) + OpenSearch híbrido/RRF/Contextual Retrieval | 10h |
| 3 | Aula 5 + Aula 6 | Avaliação (RAGAS/DeepEval/LangFuse) + Indexação avançada (Parent-Child, RAPTOR, HyDE) | 10h |
| 4 | Aula 7 + Aula 8 | Query Enhancement (Multi-Query, RAG-Fusion) + Self-RAG/CRAG/LangGraph | 10h |
| 5 | Aula 9 + Aula 10 | Graph RAG (LightRAG) + Agentic/Adaptive RAG | 10h |
| 6 | Aula 11 + Aula 12 | Técnicas complementares (Multimodal, ColBERT, Compressão, Time-Aware, DSPy) + Projeto Final | 10h |

Se seu objetivo é **implementar logo**, uma rota mais rápida e ainda coerente é: Aula 1 (skim) → Aula 2 → Aula 4 (§ busca híbrida) → Aula 3 (reranking) → Aula 5 (avaliação) → escolher 2-3 técnicas das Aulas 6-11 conforme seu problema → Aula 12 como esqueleto de arquitetura.

---

## AULA 1 — Fundamentos: NLP, Embeddings, Transformers, LLMs e o "porquê" do RAG

### 1.1 A escada evolutiva da representação de texto

Todo o curso é uma resposta cada vez mais sofisticada à pergunta "como fazer o computador entender texto". A Aula 1 percorre essa escada evolutiva, e entender cada degrau é o que permite saber **quando** usar uma técnica mais simples (mais rápida, mais barata, mais auditável) em vez de uma mais sofisticada:

```
Bag-of-Words / TF-IDF  →  Word2Vec/GloVe/fastText  →  BERT/Sentence-Transformers/BGE-M3  →  Transformers completos  →  LLMs (GPT/Llama/Mistral)
(contagem de palavras)    (vetores densos estáticos)   (vetores densos contextuais)         (arquitetura geral)        (geração em escala)
```

**TF-IDF** (`TF(termo,doc) × log(N/DF(termo))`) pondera termos raros no corpus mas frequentes no documento. Continua relevante hoje porque é **determinístico, rápido e auditável** — em domínios regulados (jurídico, saúde) essa auditabilidade tem valor próprio, e é por isso que BM25 (evolução probabilística do TF-IDF) volta como um dos dois "braços" da busca híbrida na Aula 4. Não descarte TF-IDF/BM25 como "técnica velha" — ele é peça estrutural do pipeline final.

**Word2Vec / GloVe / fastText**: primeira geração de embeddings densos (100-300 dims). Mecanismo do Word2Vec: uma rede neural rasa treinada para prever palavra↔contexto (CBOW/Skip-gram); os *pesos aprendidos* (não a tarefa de previsão em si) viram os embeddings. Propriedade notável: aritmética vetorial preserva relações semânticas (rei - homem + mulher ≈ rainha). **Limitação fundamental**: um único vetor por palavra, independente de contexto — "banco" (financeiro) e "banco" (assento) têm o mesmo vetor. fastText resolve parcialmente o problema de vocabulário fora do dicionário (OOV) ao representar palavras como soma de n-gramas de caracteres, o que é ótimo para erros de digitação/variações morfológicas.

**BERT / Sentence-Transformers / BGE-M3**: a virada contextual. BERT usa **atenção** (Aula 1 §5) para gerar um vetor diferente para a mesma palavra dependendo da frase inteira. Foi pré-treinado com Masked Language Modeling (prever token mascarado) + Next Sentence Prediction. Mas BERT sozinho não serve para comparar textos em escala (é um *cross-encoder* implícito — caro). **Sentence-Transformers** resolve isso com uma arquitetura *siamesa* (dois encoders com pesos compartilhados) treinada para que frases similares gerem embeddings próximos — isso transforma BERT num *bi-encoder*, viabilizando busca vetorial em milhões de documentos (cada documento é codificado uma única vez, offline). **BGE-M3** é o modelo padrão do curso: multilíngue (100+ idiomas), contexto de 8192 tokens, e — este é o diferencial mais importante para engenharia — **três modos num único modelo**: denso (semântico), esparso (tipo BM25 aprendido) e multi-vetor (estilo ColBERT, ver Aula 11). Isso é o que permite, mais adiante, montar busca híbrida com um único modelo de embedding.

**Por que isso importa para você construir soluções novas:** ao escolher um modelo de embedding para um projeto, a pergunta não é "qual é o melhor modelo" e sim "qual granularidade de contexto e qual modo de busca (denso/esparso/multi-vetor) meu problema precisa" — essa é uma decisão de engenharia recorrente em todo o curso.

### 1.2 Arquitetura Transformer — o mecanismo que sustenta tudo depois

Este é o tópico mais denso matematicamente e vale reler até ficar intuitivo, porque **todo o resto do curso (LLMs, rerankers, agentes) é uma aplicação de decoder ou encoder Transformer**.

- **Atenção (Q, K, V):** para cada token, três projeções lineares aprendidas geram *Query* ("o que procuro"), *Key* ("o que ofereço") e *Value* ("o que contribuo"). Score de atenção = `softmax(QKᵀ / √d_k) · V`. O `√d_k` estabiliza a escala do produto escalar. **Multi-head** roda esse processo em paralelo (8-16 cabeças) com matrizes distintas, cada cabeça aprendendo um tipo diferente de relação (sintática, correferência, etc.).
- **Encoder vs Decoder vs Encoder-Decoder:** encoder = atenção bidirecional (vê a frase toda) → bom para *compreensão* (BERT, BGE-M3, rerankers). Decoder = atenção causal (só vê tokens anteriores) → bom para *geração* (GPT, Llama, Mistral). Encoder-Decoder (T5, BART) → tradução/sumarização estruturada.
- **Positional Encoding:** sem ele, o Transformer (que processa tudo em paralelo) não tem noção de ordem — "o réu matou a vítima" ≡ "a vítima matou o réu". RoPE/ALiBi (usados em LLMs modernos) permitem generalizar para sequências mais longas que as vistas em treino.
- **Por que LLMs "alucinam" — consequência direta da arquitetura decoder:** o decoder é treinado para prever o **próximo token mais plausível**, não o mais verdadeiro. Não existe, na arquitetura, um mecanismo de verificação factual. Isso **não é um bug corrigível com prompt engineering sozinho** — é a razão estrutural de existir o RAG.

### 1.3 LLMs e parâmetros de geração

Modelos do curso: Llama 3.1 (8B/70B/405B, contexto 128k), Mistral 7B (janela deslizante eficiente), Mixtral 8x7B (Mixture-of-Experts: ativa só 2 de 8 "especialistas" por token → qualidade de 47B ao custo de 13B). Infra local: **Ollama** (`ollama serve`, API REST + endpoint OpenAI-compatible em `/v1`), escolhido por rodar em Windows/Mac/Linux sem CUDA. Em produção com GPU dedicada, o material aponta **vLLM** (PagedAttention, alto throughput) como alternativa — a portabilidade entre os dois é feita trocando apenas `base_url`.

Parâmetros-chave para engenharia de prompt:
- **Temperature** (0-2): 0 = determinístico (bom para extração factual/conclusão jurídica); 0.7-1.0 = criativo (bom para gerar variações/argumentos alternativos).
- **Top-p (nucleus sampling)**: restringe amostragem ao menor conjunto de tokens cuja probabilidade acumulada ≥ p.
- **Quantização** (float32→int8/int4): reduz VRAM até 4x com perda de qualidade geralmente pequena — é o que torna modelos de 70B rodáveis em hardware acessível.

### 1.4 Por que RAG existe — os 3 problemas que ele resolve

1. **Knowledge cutoff**: o modelo não sabe de nada após a data de treino.
2. **Alucinação**: o modelo gera texto *plausível*, não necessariamente *verdadeiro* — problema estrutural do decoder (ver 1.2).
3. **Fine-tuning não é solução geral**: caro, estático (cria um novo cutoff), sofre de *catastrophic forgetting*, e **não elimina alucinação** (o modelo pode "saber mais" e ainda inventar).

RAG: em vez de confiar na memória paramétrica do LLM, **recupera documentos relevantes de uma base externa e os injeta como contexto** antes da geração. Fluxo canônico: query → embedding → busca vetorial → top-k documentos → prompt (contexto + pergunta) → LLM → resposta com fontes rastreáveis. Isso desacopla conhecimento (no banco de dados, atualizável em tempo real) de raciocínio (no LLM).

### 1.5 Panorama das 25 técnicas do curso (mapa mental)

```
FUNDAÇÃO (1-3): embeddings densos · FAISS/OpenSearch · chunking básico
INGESTÃO (2/extra): Docling · chunking semântico/hierárquico · metadata enrichment
RETRIEVAL AVANÇADO (4-6): híbrido+RRF · HyDE · multi-query · contextual compression · parent-child · RAPTOR
RERANKING & RACIOCÍNIO (3/7): cross-encoder · LLM rerank · step-back · RAG-Fusion
SELF-CORRECTION (8): Self-RAG · CRAG · LangGraph
GRAFOS (9): LightRAG · Graph RAG
AGENTES (10): ReAct · Agentic RAG · Adaptive RAG
COMPLEMENTARES (11): Multimodal (CLIP) · Compressão (LLMLingua) · ColBERT · Time-Aware · DSPy
AVALIAÇÃO & PRODUÇÃO (5/12): RAGAS · DeepEval · LangFuse · deploy
```

**Onde praticar:** `aula1/scripts/00-05_*.py` (ambiente, NLP básico, embeddings+similaridade, métricas de retrieval, indexação OpenSearch, RAG mínimo); `aula1/labs/LAB1_Setup_Ambiente_Completo.ipynb` e `LAB2_Embeddings_BGE_M3_UMAP.ipynb`.

---

## AULA 2 — Chunking, Docling e o pipeline Naive RAG (o baseline de tudo)

### 2.1 Por que chunking é a decisão mais impactante do pipeline

O chunk é a **unidade atômica de recuperação**: o sistema nunca busca no documento inteiro, só nos chunks indexados. Um chunk ruim (cortado no meio de uma ideia) gera um embedding incoerente, que gera retrieval irrelevante, que gera resposta errada — mesmo com o melhor modelo de embedding e o melhor LLM. Estudos citados no material mostram que trocar fixed-size por recursive chunking pode subir o precision@3 em 18-34% **sem tocar no LLM**.

Três dimensões sempre em tensão: **granularidade** (chunk pequeno = precisão mas perde contexto; chunk grande = contexto mas dilui relevância), **continuidade** (overlap evita perder informação na fronteira entre chunks) e **coerência semântica** (um chunk deve representar uma única "unidade de significado").

### 2.2 As cinco estratégias de chunking (aprenda a escolher, não decore)

| Estratégia | Mecanismo | Melhor para |
|---|---|---|
| **Fixed-size** | Corta a cada N caracteres/tokens, com overlap opcional | Textos normativos uniformes, ingestão em massa |
| **Recursive** | Tenta separadores em cascata (`\n\n` → `\n` → `. ` → `" "` → char) até o chunk caber no limite | Default robusto para texto corrido |
| **Semantic** | Embeda cada sentença, mede distância de cosseno entre sentenças adjacentes, quebra onde a distância excede um threshold (percentile/std/IQR/gradient) | Narrativas longas, laudos, relatórios sem estrutura clara |
| **Sentence-Window** | Indexa cada **sentença individual** (embedding preciso) mas guarda em metadata uma **janela** de N sentenças ao redor; na recuperação, busca pela sentença mas devolve a janela ao LLM | Pareceres densos, queries que buscam fatos pontuais (datas, valores) |
| **Document-aware (header-based)** | Usa a estrutura de headers (H1/H2/H3) do Markdown gerado pelo Docling; cada chunk herda metadados hierárquicos (ex.: seção "FUNDAMENTAÇÃO") | PDFs já estruturados; permite filtro cirúrgico por seção |

**O ponto de engenharia mais reaproveitável desta aula**: a estratégia sentence-window **desacopla o que é indexado do que é devolvido ao LLM** — você busca pela unidade mais precisa (frase) mas gera com a unidade mais rica (janela de contexto). Esse mesmo princípio de desacoplamento reaparece, ampliado, na Aula 6 (Parent-Child).

### 2.3 Docling — ingestão estruturada de documentos complexos

PDFs reais (colunas, tabelas, rodapés, OCR) quebram extratores simples como PyPDF2 (que devolve texto linear desordenado, ignorando tabelas e figuras). O pipeline interno do Docling: **PDF backend (pypdfium2)** → **Layout Analysis (DocLayNet)** detecta título/parágrafo/tabela/figura → **Table Structure Recovery** reconstrói tabela como DataFrame → **OCR (EasyOCR/Tesseract)** se o PDF for escaneado → **Reading Order Detection** corrige ordem de leitura em layouts complexos → objeto `DoclingDocument` com `.export_to_markdown()`, `.export_to_dict()`, `.tables`, `.pictures`.

Regra prática: use PyPDF2/PDFMiner para PDFs digitais simples e alto volume; use Docling quando há tabelas, colunas, necessidade de OCR ou quando você quer preservar hierarquia para chunking document-aware.

### 2.4 Pipeline Naive RAG — o baseline contra o qual tudo se compara

```
INDEXAÇÃO (offline): PDF → Docling (Markdown) → RecursiveCharacterTextSplitter (chunk_size=800, overlap=150)
                     → Documents com metadata → BGE-M3 embeddings (dim=1024) → índice kNN no OpenSearch

RETRIEVAL+GERAÇÃO (online): pergunta → BGE-M3 → busca kNN top-5 → monta contexto → prompt template
                     → LLM (Llama local via Ollama) → resposta com citação de fonte
```

Stack do curso: Docling (ingestão) → LangChain TextSplitters (chunking) → BGE-M3 via Ollama (embeddings) → OpenSearch kNN (vector store) → Ollama/Llama (geração) → LangChain LCEL (orquestração).

**Limitações documentadas do Naive RAG** (o mapa de todo o resto do curso): sem reranking (Aula 3), sem filtragem por metadados (Aula 4), sem compressão de contexto (Aula 3/11), sem expansão de query (Aula 6/7), sem verificação de fatos (Aula 8), sem memória conversacional (Aulas 9-11).

### 2.5 As 5 armadilhas mais comuns (guarde como checklist de debugging)

1. Chunk pequeno demais para textos jurídicos longos (mede o P90 do tamanho de sentença antes de escolher `chunk_size`).
2. **Modelo de embedding diferente na indexação e na query** — espaços vetoriais incompatíveis, a similaridade perde sentido. Sempre fixe o modelo como constante e reuse a mesma instância dos dois lados.
3. PyPDF2 em PDF de layout complexo → texto embaralhado.
4. Não preservar metadados de proveniência (fonte, número do processo, seção) → resposta sem citação verificável.
5. Índice kNN do OpenSearch criado com dimensão diferente do embedding gerado (erro só aparece na hora da busca).

**Onde praticar:** `aula2/scripts/01_ingestao_docling.py`, `02_chunking_comparar.py`, `03_indexar_chunks_opensearch.py`, `04_naive_rag.py`; labs `LAB1..LAB5`; datasets reais em `aula2/datasets/` (PDFs digitais e escaneados).

**Reaproveitamento em soluções novas:** este é o esqueleto de qualquer RAG que você for construir. Comece sempre por aqui, meça um baseline com RAGAS (Aula 5) e só então adicione complexidade — nunca pule direto para técnicas avançadas sem medir o ganho contra este baseline.

---

## AULA 3 — Advanced RAG: Query Rewriting, Reranking e Modularidade

### 3.1 Os três pontos de falha do Naive RAG

1. **Vocabulary gap**: usuário fala coloquial, documento fala técnico-jurídico (ex.: "prender sem mandado" vs. "prisão em flagrante").
2. **Contexto truncado**: chunking corta entidades relevantes entre fronteiras.
3. **Sem filtragem pós-retrieval**: similaridade semântica global ≠ relevância para a pergunta específica.

### 3.2 Query Rewriting — três técnicas para reduzir o vocabulary gap

- **Paraphrase rewriting**: LLM gera N reformulações técnicas da query original.
- **HyDE-lite**: LLM gera um **parágrafo hipotético** que responderia à pergunta, na linguagem do domínio; esse parágrafo (não a query) é embedado para a busca. Funciona porque o documento hipotético "mora" no mesmo espaço semântico dos documentos reais. Risco: pode alucinar detalhes no parágrafo hipotético — nunca use para queries factuais específicas (números de processo etc.).
- **Step-back**: abstrai a query para um princípio mais geral antes de buscar (ex.: "João pode apelar por erro de dosimetria?" → "Quais os fundamentos jurídicos da apelação criminal?"). Detalhado a fundo na Aula 7.

### 3.3 Reranking — bi-encoder vs cross-encoder (conceito central do curso)

Esta é uma distinção que você vai usar sempre que desenhar um pipeline de retrieval:

- **Bi-encoder** (ex.: BGE-M3): query e documento são codificados **independentemente**; a busca é uma multiplicação vetorial pré-computável (ANN). Rápido — O(1) por query com índice — mas menos preciso (sem interação entre query e doc).
- **Cross-encoder** (ex.: `BAAI/bge-reranker-v2-m3`): query e documento são processados **juntos** pelo Transformer (atenção cruzada completa). Muito mais preciso, mas caro — O(N) por query, um forward pass por par (query, doc).

**Arquitetura padrão que resulta disso, e que você deve replicar em qualquer sistema sério:**
```
1. Bi-encoder: busca rápida em 1M+ documentos → top-100 candidatos
2. Cross-encoder: reranking preciso nos top-100 → top-5 final enviados ao LLM
```
Score do cross-encoder = `sigmoid(W·h_CLS + b)`, interpretável em faixas (0.85-1.0 altamente relevante, <0.3 descartar).

### 3.4 Modular RAG — arquitetura para produção

Formaliza o RAG como módulos independentes com interface contratual (classes abstratas `BaseRetriever`, etc.), permitindo trocar retriever/reranker/gerador em runtime sem afetar o resto do pipeline — análogo à injeção de dependência em engenharia de software. Benefícios em contexto regulado: múltiplos corpora por domínio, auditoria por módulo, evolução gradual, A/B testing sem downtime.

```python
class BaseRetriever(ABC):
    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> List[dict]: ...
```

### 3.5 LangFuse — observabilidade desde já

Introduzido aqui e usado no resto do curso: **traces** (execução completa) contendo **spans** (cada etapa: query rewriting, retrieval, reranking, geração), com latência e tokens por etapa. Instrumentação via decorator `@observe()`. Métricas-alvo: latência p95 < 5s, retrieval < 500ms, reranking < 3s, taxa de erro < 0.1%.

**Onde praticar:** `aula3/scripts/01_query_rewriting.py`, `03_reranking_bge.py`, `04_advanced_rag.py`; labs `LAB1..LAB6` (inclui instrumentação LangFuse completa).

**Reaproveitamento:** o padrão bi-encoder→cross-encoder é a otimização de retrieval com melhor custo-benefício que existe — implemente-o antes de qualquer técnica mais exótica das aulas seguintes.

---

## AULA 4 — OpenSearch Completo: Busca Híbrida, RRF, Neural Sparse e Contextual Retrieval

### 4.1 Por que busca vetorial pura não basta

Embeddings densos falham em: termos técnicos raros/siglas (pouco representados no pré-treino), match exato de números de lei/súmula, e vocabulário fora-do-vocabulário (OOV). BM25 (lexical) resolve exatamente esses casos, mas falha em sinonímia. **Busca híbrida = BM25 + embeddings**, o melhor dos dois mundos.

### 4.2 Arquitetura de índice híbrido no OpenSearch

Um documento carrega simultaneamente um campo `text` (BM25/Lucene) e um campo `knn_vector` (dimensão = dimensão do embedding, engine `faiss`, `space_type: cosinesimil`). A query `hybrid` executa as duas sub-queries em paralelo; um **search pipeline** com `normalization-processor` normaliza e combina os scores (que, sem isso, são incomparáveis: BM25 varia 0-20, cosine 0-1).

### 4.3 Reciprocal Rank Fusion (RRF) — o algoritmo mais reaproveitável da aula

RRF combina **posições (ranks)**, não scores brutos — por isso é robusto a escalas incompatíveis e a outliers:

```
RRF(d) = Σ  1 / (k + rank_r(d))     (soma sobre cada lista de ranking r; k=60 tipicamente)
```

Documentos que aparecem bem posicionados em **múltiplas listas** são favorecidos sobre um documento excepcional em apenas uma lista — exatamente o comportamento desejado ao fundir buscas semânticas e lexicais (e, na Aula 7, ao fundir múltiplas sub-queries). **Esse mesmo algoritmo RRF reaparece três vezes no curso**: fusão BM25+kNN (aqui), fusão de rankings do RAG-Fusion (Aula 7) e, implicitamente, em qualquer combinação de múltiplos retrievers.

Pesos da combinação (`weights: [w_bm25, w_knn]`): para queries factuais (artigo X da lei Y) pese mais o BM25 (ex. 0.7/0.3); para queries semânticas, pese mais o kNN (0.3/0.7).

### 4.4 Neural Sparse Search (SPLADE)

Terceira via entre denso e lexical: um modelo neural (baseado em BERT) gera um **vetor esparso** onde cada dimensão corresponde a um termo do vocabulário, com expansão implícita de termos relacionados (ex.: "veículo" ganha peso também em "carro"). Vantagem: **interpretabilidade** (você vê exatamente quais termos pesaram) — importante para auditoria em domínios regulados. Armazenado no OpenSearch com o tipo `rank_features`.

### 4.5 Contextual Retrieval (técnica da Anthropic, 2024)

Resolve o problema de chunks que, isolados, perdem todo contexto situacional (ex.: "O Tribunal negou provimento ao recurso" sozinho não diz de qual caso). Antes de indexar, um LLM gera uma frase de contexto situacional para cada chunk (usando o documento completo como referência), e essa frase é **prependada** ao chunk antes do embedding: `[CONTEXTO: ...] + chunk_original`. Ganhos reportados: ~35% em Context Precision. Custo: uma chamada de LLM por chunk na ingestão (mais lento, mas só acontece uma vez, offline).

### 4.6 Métricas de avaliação de retrieval (pré-requisito para a Aula 5)

- **MRR** (Mean Reciprocal Rank) = média de `1/rank` do primeiro resultado relevante por query.
- **Recall@K** = proporção de documentos relevantes capturados nos top-K.
- **NDCG@K** = pondera relevância graduada e posição: `DCG@K = Σ rel_i/log2(i+1)`, normalizado pelo DCG ideal.

**Onde praticar:** `aula4/scripts/01_indexar_hibrido.py`, `02_busca_hibrida_rrf.py`, `03_contextual_retrieval.py`, `06_neural_sparse.py`; labs `LAB1..LAB6`.

**Reaproveitamento:** busca híbrida + RRF é, junto com bi-encoder→cross-encoder (Aula 3), a dupla de técnicas com maior retorno prático em qualquer RAG de produção — comece por essas duas antes de recorrer a agentes ou grafos.

---

## AULA 5 — Avaliação e Observabilidade: RAGAS, DeepEval, LangFuse Avançado

### 5.1 O problema: como saber se o RAG está funcionando de verdade

Métricas clássicas de NLP (BLEU/ROUGE medem overlap de n-gramas, perplexidade mede fluência, não factualidade) não servem para RAG. A resposta do curso é **LLM-as-Judge**: um LLM avalia se a resposta está fundamentada no contexto recuperado — abordagem do RAGAS, com alta correlação com avaliação humana e escalável.

### 5.2 As 4 métricas RAGAS (decore os diagnósticos, não só as fórmulas)

| Métrica | Mede | Fórmula conceitual | Meta | Se estiver baixa... |
|---|---|---|---|---|
| **Faithfulness** | Resposta é fundamentada no contexto? | nº afirmações suportadas / nº total de afirmações (LLM decompõe a resposta em afirmações atômicas e verifica cada uma) | ≥ 0.80 | LLM alucinando além do contexto → reforce o prompt ("responda só com base nos trechos") |
| **Answer Relevancy** | Resposta é pertinente à pergunta? | similaridade entre a pergunta original e perguntas sintéticas geradas a partir da resposta | ≥ 0.75 | Resposta vaga/desviada → revisar prompt |
| **Context Recall** | O ground-truth pode ser derivado dos contextos? (mede o retriever) | nº sentenças do ground-truth atribuíveis ao contexto / total | ≥ 0.70 | Retriever perdendo documentos → revisar chunking/k/embedding |
| **Context Precision** | Os contextos recuperados são relevantes? (mede ruído do retriever) | precisão ponderada por posição no top-k | ≥ 0.70 | Retriever trazendo ruído → adicionar reranking, ajustar alpha da busca híbrida |

Este quadro de diagnóstico (qual métrica aponta para qual componente do pipeline) é a ferramenta mais valiosa da aula — é o que transforma "a resposta está ruim" em "o problema está no retriever, especificamente em recall".

### 5.3 DeepEval — testes unitários (a diferença de "média" para "assert")

RAGAS mede qualidade **média** num dataset; DeepEval roda **assertions por caso individual**, integradas a pytest/CI-CD — analogia: RAGAS é relatório de cobertura, DeepEval é teste unitário que bloqueia deploy. Métricas usadas: `FaithfulnessMetric`, `AnswerRelevancyMetric`, `HallucinationMetric`, `ToxicityMetric`, `BiasMetric` (as duas últimas são especialmente relevantes quando o corpus vem de fontes com linguagem historicamente enviesada, como boletins de ocorrência).

### 5.4 LangFuse Scores API — de rastreamento a monitoramento contínuo

Permite anexar scores numéricos (RAGAS, DeepEval ou feedback humano) a qualquer trace, transformando o LangFuse em painel de qualidade em produção, com alertas quando métricas caem abaixo da meta. **Insight-chave**: um pipeline que passa em todos os testes hoje pode degradar silenciosamente quando o corpus muda (nova lei, novo acórdão) — só o monitoramento contínuo pega isso.

### 5.5 Cadência de avaliação recomendada

| Momento | Ferramenta | Frequência |
|---|---|---|
| Desenvolvimento | RAGAS + Pandas | a cada mudança de pipeline |
| Antes do deploy | DeepEval + pytest | CI/CD |
| Produção | LangFuse Scores API | por requisição (amostragem) |
| Manutenção | LangFuse dashboard + RAGAS | semanal |

**Onde praticar:** `aula5/scripts/03_ragas_avaliar.py`, `04_deepeval_testes.py`, `05_langfuse_scores.py`; labs `LAB1..LAB7`.

**Reaproveitamento:** monte esse pipeline de avaliação **antes** de otimizar qualquer coisa — sem baseline medido, você não consegue provar que uma técnica avançada (Aulas 6-11) realmente ajudou. Essa é, aliás, uma armadilha explicitamente citada na Aula 6 ("nunca assuma que uma técnica avançada é melhor sem medir").

---

## AULA 6 — Indexação Avançada: Parent-Child, RAPTOR e HyDE

### 6.1 O dilema do chunking plano em documentos longos

Um único tamanho de chunk serve mal a dois tipos de pergunta: pontual (exige chunk pequeno e preciso) e abrangente (exige contexto amplo, de múltiplas seções). Solução conceitual comum às três técnicas desta aula: **separar a unidade de busca da unidade de contexto/geração** — o mesmo princípio da estratégia sentence-window da Aula 2, agora levado a duas escalas maiores: dentro do documento (Parent-Child) e entre documentos (RAPTOR).

### 6.2 Parent-Child (Hierarchical Indexing)

Dois tamanhos de chunk com papéis distintos: **chunks filho** (pequenos, ex. 128 tokens) são indexados no vector store para busca precisa; **chunks pai** (grandes, ex. 512 tokens) ficam num docstore e são recuperados quando um filho tem match, fornecendo contexto completo ao LLM.

```
Query → embedding → busca entre FILHOS → match no filho F-002b → recupera o PAI-002 correspondente → LLM recebe o PAI
```

Implementação de referência: `HierarchicalNodeParser.from_defaults(chunk_sizes=[512, 128])` do LlamaIndex + `AutoMergingRetriever` (que promove automaticamente o pai quando uma fração configurável dos filhos — ex. 30% — é recuperada). **Armadilha:** a razão pai:filho precisa ser grande (1:4 ou 1:8) — com 1:2 o auto-merging quase nunca ativa.

Aplicação jurídica natural: artigo de lei (pai) com incisos/parágrafos (filhos) — uma busca por um inciso específico recupera o artigo inteiro como contexto.

### 6.3 RAPTOR — árvore de abstrações sobre o corpus inteiro

Resolve perguntas de **síntese entre documentos** (ex. "quais as tendências jurisprudenciais sobre X nos últimos 5 anos"), que nenhum chunk individual consegue responder porque exigem agregação de múltiplos documentos. Algoritmo:

1. Gera embeddings de todos os chunks.
2. **Reduz dimensionalidade com UMAP** (de 1024 para ~10 dims) — necessário porque clustering em alta dimensão sofre da "maldição da dimensionalidade".
3. **Clusteriza com GMM** (Gaussian Mixture Model, não k-means) — permite que um chunk pertença, probabilisticamente, a múltiplos clusters (ex. um acórdão sobre privacidade *e* criptomoedas). Número de clusters escolhido por BIC.
4. **Sumariza cada cluster com um LLM**, gerando um novo "chunk" de nível superior.
5. **Recursão**: os resumos de nível 1 alimentam o clustering de nível 2, e assim sucessivamente, até formar uma árvore (folhas = chunks originais; raiz = síntese geral do corpus).

Duas estratégias de query: **Collapsed Tree** (busca em todos os níveis simultaneamente — mais simples, cobre detalhe e visão geral) e **Tree Traversal** (desce nível a nível escolhendo o ramo mais similar — mais controlado). **Armadilha:** RAPTOR exige corpus grande (50-100+ chunks por cluster) para gerar clusters coerentes; com poucos documentos, os clusters ficam artificiais.

### 6.4 HyDE (Hypothetical Document Embeddings) — versão completa

Já introduzido de forma simplificada na Aula 3; aqui, o mecanismo completo: o gap semântico entre query coloquial e documento técnico faz a similaridade de cosseno cair (ex. 0.41). HyDE pede ao LLM para gerar um **documento hipotético** que responderia à pergunta, na linguagem formal esperada, e embeda esse documento (não a query) para a busca — a similaridade sobe (ex. para 0.86) porque o hipotético "mora" no mesmo espaço semântico dos documentos reais. **Funciona geometricamente, não magicamente**: você está trocando o vetor de busca por um vetor mais próximo da distribuição do corpus.

**Quando HyDE falha:** queries sobre fatos específicos (o LLM pode alucinar números de processo no hipotético), corpus pequeno (custo não compensa), domínios muito especializados sem dados de pré-treino (hipotéticos imprecisos), queries ambíguas (hipotético pode ir na direção errada — nesse caso combine com Multi-Query, Aula 7).

### 6.5 Árvore de decisão da aula

```
Query pontual sobre artigo/cláusula + corpus estruturado           → Parent-Child
Query de síntese/tendência + corpus grande (>100 docs)              → RAPTOR
Gap semântico usuário leigo vs. corpus técnico                      → HyDE
Combinação dos problemas                                             → combine as técnicas (não são excludentes)
```

**Onde praticar:** `aula6/scripts/02_parent_child.py`, `03_raptor.py`, `04_hyde.py`, `05_comparar_tecnicas.py`; labs `LAB1..LAB4`.

**Reaproveitamento:** Parent-Child é a técnica de maior retorno/esforço desta aula para qualquer corpus com estrutura hierárquica clara (leis, contratos, manuais). RAPTOR só vale o custo de indexação (alto) se você realmente tiver perguntas de síntese multi-documento — não implemente por modismo.

---

## AULA 7 — Query Enhancement: Multi-Query, Step-Back, Decomposition e RAG-Fusion

### 7.1 Vocabulary mismatch, revisitado com profundidade

A mesma intenção do usuário pode ser expressa de formas completamente diferentes do vocabulário indexado (ex. "demitir funcionário" vs. "rescisão contratual"). A família de soluções desta aula não tenta mudar o usuário — **enriquece a query antes do retrieval**, com três estratégias que atacam facetas diferentes do problema.

### 7.2 Multi-Query RAG

Gera N variações semânticas da query original com um LLM (ex. cobrindo terminologia técnica, coloquial, doutrinária e processual), executa retrieval para cada uma, deduplica e funde os resultados.

```python
retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(), llm=llm)
```

**Escolha de N**: N=1 = Naive RAG. N=3-5 é a faixa ótima empírica (ganho marginal decresce depois de N=4). N≥8 arrisca *query drift* (variações se afastam da intenção original, reduzindo precisão sem ganho real de recall).

**Deduplicação**: a deduplicação por igualdade exata de texto falha quando o mesmo conteúdo foi indexado com pequenas variações (overlap de chunking). A solução robusta é deduplicação por **similaridade semântica** (remove documentos com cosseno > threshold, tipicamente 0.85, usando um algoritmo guloso).

### 7.3 Step-Back Prompting (mecanismo completo, complementa a Aula 3)

Para queries hiper-específicas (nomes, datas, artigos exatos que provavelmente não existem *verbatim* no corpus), pergunte primeiro pelo **princípio geral**, recupere documentos sobre esse princípio, e só então gere a resposta específica usando esse contexto + a query original. **Quando NÃO ajuda:** queries já abstratas (risco de sobre-generalizar), buscas por entidade específica (a abstração "perde o alvo"), corpus já no nível de princípios gerais.

### 7.4 Query Decomposition

Perguntas compostas (ex. "diferenças entre prisão preventiva e temporária, quando cada uma se aplica e quais os direitos do preso") são, na prática, várias perguntas atômicas independentes. Decompor com um LLM em sub-perguntas e tratar cada uma separadamente evita que um único retrieval tente (e falhe em) cobrir todas as facetas.

### 7.5 RAG-Fusion — Multi-Query + RRF (o ápice desta família)

Combina geração de sub-queries com o algoritmo **RRF** (o mesmo da Aula 4, `RRF(d) = Σ 1/(k+rank)`) para fundir os rankings de cada sub-query, em vez de só deduplicar. Isso premia documentos **consistentemente relevantes em múltiplos rankings** sobre documentos excepcionais em apenas um. Implementação real exige **paralelismo assíncrono** (`asyncio.gather`) para que N sub-queries não multipliquem a latência.

### 7.6 Trade-off Recall × Latência × Custo (raciocínio de engenharia, não só teoria)

Recall cresce com N mas com retorno marginal decrescente (grande ganho de N=1→3, pequeno de N=3→5, desprezível acima de 5). Latência: com asyncio real, domina a query mais lenta (quase constante com N); sem asyncio, cresce linearmente. Custo: cada sub-query consome tokens de geração e de embedding — crítico se você usa API paga, irrelevante (exceto energia/GPU) se você usa LLM local.

### 7.7 Guia de decisão consolidado

```
Usuário usa linguagem técnica? → SIM: busca híbrida (Aula 4) já basta
                                → NÃO: query enhancement necessário
   Query hiper-específica (nome/data/artigo)?  → Step-Back (+ Multi-Query opcional)
   Query multi-facetada?                        → Query Decomposition + RAG-Fusion
   Caso geral                                   → Multi-Query RAG (N=3-4)
```

**Onde praticar:** `aula7/scripts/02_multi_query.py`, `03_step_back.py`, `04_rag_fusion.py`, `05_benchmark.py`; labs `LAB1..LAB6` (inclui Langflow).

**Reaproveitamento:** RAG-Fusion com N=3 e asyncio é, na prática, o "upgrade" de maior ROI sobre um RAG híbrido já funcionando, porque reaproveita o mesmo RRF que você já implementou na Aula 4 — é literalmente aplicar a mesma fórmula em outra camada do pipeline.

---

## AULA 8 — Self-RAG, CRAG e LangGraph: Auto-correção e Orquestração Condicional

### 8.1 As três limitações do RAG convencional que esta aula ataca

Recuperação indiscriminada (busca mesmo quando desnecessário), sem verificação de relevância (usa documentos recuperados cegamente), sem autocorreção (retrieval ruim → geração ruim, sem detecção).

### 8.2 Self-RAG — o modelo aprende a se autoavaliar (requer fine-tuning)

Introduz 4 tokens de controle que o próprio LLM emite durante a geração:

| Token | Função | Valores |
|---|---|---|
| `[Retrieve]` | Decide se busca documentos | yes/no |
| `[ISREL]` | Avalia relevância do documento recuperado | relevant/irrelevant |
| `[ISSUP]` | Verifica suporte factual da geração no documento | fully/partially/no support |
| `[ISUSE]` | Avalia utilidade da resposta | 1-5 |

Fluxo: o modelo decide autonomamente `[Retrieve=yes/no]`; se recupera, avalia `[ISREL]`; gera a resposta; verifica `[ISSUP]` (a resposta bate com o documento?); avalia `[ISUSE]`. **Limitação crítica de engenharia**: exige um modelo **especificamente treinado** (fine-tuned) para emitir esses tokens — não funciona com um LLM genérico "fora da caixa". Isso limita adoção prática fora de modelos específicos (ex. `llama-2-7b-selfrag`).

### 8.3 CRAG — correção sem fine-tuning (training-free), o caminho mais reaproveitável

Um **avaliador de qualidade** (LLM-as-Judge, prompt simples pedindo um score 0-1 de relevância) examina os documentos recuperados **antes** da geração e roteia:

```
score ≥ 0.7        → usa documentos locais
0.3 ≤ score < 0.7  → combina documentos locais + busca web (Tavily)
score < 0.3        → descarta locais, usa só busca web
```

Isso é essencialmente um **CRAG genérico** que você pode implementar com qualquer LLM, sem treinar nada — só precisa de um prompt de avaliação e um roteamento condicional.

### 8.4 LangGraph — a ferramenta de orquestração que viabiliza CRAG (e agentes na Aula 10)

Modela o pipeline como uma **máquina de estados**: `StateGraph` define o schema do estado (um `TypedDict` compartilhado entre nós), **nós** são funções Python que leem/atualizam o estado, e **arestas condicionais** são funções que decidem o próximo nó com base no estado atual (ex. rotear por `score` como no CRAG). Diferencial-chave: suporta **ciclos** (voltar a um nó anterior — ex. reescrever a query se o retrieval falhar repetidamente), sempre com um limite máximo de iterações para não entrar em loop infinito.

```python
class GraphState(TypedDict):
    question: str; documents: List[str]; web_results: List[str]
    generation: str; relevance_score: float

def route_documents(state: GraphState) -> str:
    return "generate" if state["relevance_score"] >= 0.5 else "web_search"
```

### 8.5 Tavily — busca web pensada para consumo por LLM

API de busca que já devolve resultados resumidos/estruturados (não HTML bruto), usada como fallback quando o avaliador do CRAG considera os documentos locais insuficientes.

### 8.6 Comparativo e guia de decisão

| Critério | Advanced RAG | Self-RAG | CRAG |
|---|---|---|---|
| Fine-tuning | Não | **Sim** | Não |
| Quando recupera | Sempre | Sob demanda (token) | Sempre, mas avalia qualidade |
| Fallback automático | Não | Não | **Sim (web)** |
| Melhor caso | Corpus grande e estável | Alta precisão factual | Corpus incompleto/desatualizado |

**Onde praticar:** `aula8/scripts/02_self_rag.py`, `03_avaliador.py`, `04_crag.py`; labs `LAB1_Self_RAG_Ollama/vLLM.ipynb`, `LAB2_CRAG_LangGraph.ipynb`, `LAB4_Tavily_Integracao.ipynb`.

**Reaproveitamento:** CRAG + LangGraph é o padrão de auto-correção mais acessível para produção (não exige modelo especial); guarde o padrão "avaliador LLM-as-judge → roteamento condicional → fallback" — ele reaparece, generalizado, no Adaptive RAG da Aula 10.

---

## AULA 9 — Graph RAG com LightRAG: Raciocínio Multi-hop sobre Grafos de Conhecimento

### 9.1 O problema que RAG textual não resolve: raciocínio multi-hop

RAG convencional trata chunks como unidades independentes. Perguntas que exigem **conectar entidades através de relações** (ex. "quais as conexões entre o réu Silva e organizações criminosas investigadas nos últimos dois anos") não são respondidas por chunks isolados — exigem navegar: `Silva → investigado_por → MPF`, `Silva → associado_a → Organização_Alpha`, etc.

### 9.2 Grafo de conhecimento: conceitos e construção automática

Um Knowledge Graph representa entidades (nós) e relações (arestas) como triplas `(Sujeito) → [Relação] → (Objeto)`. O **LightRAG** automatiza a extração via LLM:

```
Documento → Chunking → LLM extrai entidades+relações → entidades vão para índice vetorial (kNN)
          → relações formam grafo de adjacência → deduplicação/merge de entidades
          → detecção de comunidades (algoritmo de Leiden) → LLM sumariza cada comunidade
```

### 9.3 Quatro modos de query (a parte mais reaproveitável para engenharia)

| Modo | Mecanismo | Uso |
|---|---|---|
| **naive** | busca vetorial simples, ignora o grafo | baseline |
| **local** | foca numa entidade e seus vizinhos imediatos no grafo | "quem são os coautores investigados com Silva?" (pergunta sobre entidade específica) |
| **global** | usa sumários de **comunidades** (clusters temáticos) | "quais os principais temas do acervo?" (visão panorâmica) |
| **hybrid** (recomendado) | combina local + global + chunks textuais brutos | produção geral — melhor qualidade |

### 9.4 Stack on-premise e motivação de privacidade

vLLM (LLM local) + OpenSearch (backend unificado para os 4 tipos de armazenamento do LightRAG: docs, cache, vetores de entidades, grafo) + embeddings locais BGE-M3. Motivação explícita: dados de investigação/LGPD não podem trafegar para APIs externas — o mesmo argumento de soberania de dados da Aula 1, agora aplicado a um componente novo (o grafo).

### 9.5 Quando usar Graph RAG vs. RAG textual

Use grafo quando: mapeamento de redes (conexões entre pessoas/organizações), rastreamento de citações jurisprudenciais em cadeia, análise de fluxo financeiro, identificação de padrões temporais. Mantenha RAG textual quando: pergunta pontual e factual, corpus pequeno/atualizado com frequência, ou restrição forte de custo computacional (a extração de grafo via LLM é cara na indexação).

**Onde praticar:** `aula9/scripts/01_indexar_grafo.py`, `02_consultar_modos.py`, `03_investigacao.py`, `04_explorar_grafo.py`, `05_visualizar_grafo.py`; labs `lab1_lightrag_opensearch_vllm.ipynb`, `lab2_queries_avancadas_investigacao.ipynb`, `lab3_pipeline_juridico_reranking.ipynb`.

**Reaproveitamento:** se seu problema envolver entidades e relações explícitas (organogramas, redes sociais, cadeias de citação, fluxos financeiros), Graph RAG é a única técnica do curso que resolve isso nativamente — nenhuma quantidade de reranking ou query enhancement em RAG textual substitui essa capacidade.

---

## AULA 10 — Agentic RAG e Adaptive RAG: o LLM como Orquestrador

### 10.1 Do RAG linear ao RAG agêntico

Até aqui (mesmo em CRAG/Self-RAG/Graph RAG), o **plano** de execução é predefinido pelo desenvolvedor — o sistema decide *como* recuperar, mas não *se* precisa de mais uma rodada, nem *quais* ferramentas usar. Agentic RAG inverte isso: o **LLM decide em tempo de execução** quais ferramentas usar, em que ordem e quantas vezes, até reunir informação suficiente.

### 10.2 ReAct — Reason + Act, o padrão que sustenta agentes modernos

Ciclo: `Thought → Action → Observation → Thought → ... → Answer`. Na prática moderna, isso é implementado via **tool calling nativo** do LLM (o modelo emite uma chamada de função estruturada — nome + argumentos JSON — e o runtime executa e devolve a observação como mensagem).

```
Thought 1: Preciso dos acórdãos do acervo sobre X.
Action 1: buscar_documentos("X")
Observation 1: [3 acórdãos]
Thought 2: Falta orientação recente que pode não estar no acervo.
Action 2: buscar_web("X 2025")
Observation 2: [notícias]
Thought 3: Tenho local + atual; posso responder.
```

### 10.3 Design de ferramentas (tools) — o fator crítico de qualidade do agente

A qualidade do agente depende **diretamente** da descrição da ferramenta, porque é por ela que o LLM decide quando usar cada uma:

```
❌ RUIM: "Busca documentos"
✅ BOM:  "Busca legislação/jurisprudência no banco vetorial local. Use quando a pergunta
         envolver leis, artigos, decisões. NÃO use para fatos recentes (use buscar_web).
         Args: query (str)."
```

Regras: nome claro, descrição que diz **quando usar e quando NÃO usar**, parâmetros tipados (JSON schema).

### 10.4 Adaptive RAG — roteamento por complexidade (o ganho de custo/latência mais direto do curso)

Nem toda pergunta precisa da mesma estratégia. Um classificador (LLM) estima a complexidade e roteia para um de três caminhos:

| Caminho | Quando | Custo/Latência |
|---|---|---|
| Sem retrieval | conhecimento geral ("o que é habeas corpus?") | mínimo / <1s |
| Single-step RAG | factual, uma busca resolve | baixo / 2-5s |
| Multi-step (agente) | comparativa/multi-fonte | alto / 10-30s |

Implementação: classificador LLM (`sem_retrieval | simples | complexa`) → roteador condicional → caminho correspondente. O ganho é que **você só paga o caminho caro quando a pergunta exige** — é o mesmo princípio de roteamento condicional do CRAG (Aula 8), aplicado agora à escolha entre RAG simples e agente completo.

### 10.5 Riscos de agentes e mitigação (checklist obrigatório antes de produção)

| Risco | Mitigação |
|---|---|
| Loop infinito | limite de passos (`max_agent_steps`) |
| Tool call excessivo/redundante | memória de observações / deduplicação |
| Custo imprevisível | orçamento + alertas + caminho adaptativo |
| Alucinação de ferramenta (LLM inventa nome de tool) | validação de schema (runtime rejeita) |
| Timeout de ferramenta lenta | timeout por tool + fallback |

Em domínio regulado, soma-se a exigência de **rastreabilidade**: cada afirmação precisa de origem verificável (qual ferramenta, qual documento) — os traces do LangFuse cobrem isso.

### 10.6 Avaliação de agentes vai além da resposta final

Além de RAGAS (Faithfulness, Answer Relevancy), agentes exigem avaliar a **trajetória**: o agente escolheu as ferramentas certas, e só elas? "Recall de ferramentas" é tão importante quanto a resposta. Custo de produção se estima como: nº médio de passos × tokens por passo × preço do modelo — essa conta é o que justifica investir em Adaptive RAG.

**Onde praticar:** `aula10/scripts/02_react_manual.py`, `03_agente_ferramentas.py`, `04_adaptive_rag.py`, `05_avaliar_custo.py`; labs `lab1_agentic_rag_3_ferramentas.ipynb`, `lab2_adaptive_rag_3_caminhos.ipynb`, `lab3_avaliacao_ragas_langfuse.ipynb`.

**Reaproveitamento:** o padrão "classificador de complexidade + roteamento condicional + caminho caro só quando necessário" é diretamente aplicável a qualquer sistema com custo de LLM não-trivial — é provavelmente a técnica de maior ROI em produção real depois da busca híbrida.

---

## AULA 11 — Técnicas Complementares: Multimodal, Compressão, ColBERT, Time-Aware, DSPy

Estas cinco técnicas são independentes entre si e combináveis — trate esta aula como uma "caixa de ferramentas" a acionar conforme o problema específico, não como uma sequência.

### 11.1 Multimodal RAG (CLIP)

Documentos reais (laudos, BOs, relatórios) contêm tabelas, gráficos, plantas, imagens — RAG textual ignora tudo isso. **CLIP** projeta texto e imagem no **mesmo espaço vetorial** (treinado contrastivamente em pares imagem-legenda), permitindo `score(query_texto, imagem) = cos(CLIP_text(query), CLIP_image(imagem))`. Pipeline: Docling extrai texto/tabelas/imagens → texto embedado normalmente, imagens embedadas via CLIP → índice único → busca textual recupera imagens relevantes por similaridade.

### 11.2 Compressão de contexto (LLMLingua)

RAG ingênuo concatena todos os chunks recuperados → custo alto, latência, e o fenômeno **"lost in the middle"** (LLMs ignoram informação no meio de contextos longos). LLMLingua faz **token pruning por perplexidade**: um modelo pequeno pontua cada token; tokens previsíveis (baixa perplexidade: "de", "que", "aos") são removidos, preservando o núcleo semântico. LLMLingua-2 usa um classificador BERT para isso (mais rápido, multilíngue). Ganho típico: 2-5x menos tokens de entrada. Alternativa **abstractiva** (em vez de remover, resume): RECOMP.

### 11.3 ColBERT / Late Interaction — o meio-termo entre bi-encoder e cross-encoder

Bi-encoders comprimem o documento inteiro num único vetor — eficiente, mas termos irrelevantes "diluem" a representação. ColBERT guarda **um vetor por token** e calcula relevância via **MaxSim**: para cada token da query, pega a maior similaridade com qualquer token do documento, e soma:

```
Score = Σ_query  max_documento  cos(token_q, token_d)
```

Isso entrega precisão próxima de cross-encoders com velocidade próxima de bi-encoders (o motor PLAID comprime os vetores via clustering para viabilizar isso em escala). Custo: índice 3-5x maior. Biblioteca de referência: **RAGatouille** (encapsula ColBERTv2 numa API simples `index`/`search`). Tabela de referência do material: BM25 (nDCG@10 ~0.45-0.55, ~5ms) < Bi-encoder (~0.60-0.70, ~10ms) < ColBERT (~0.70-0.80, ~30ms) < Cross-encoder (~0.75-0.85, ~200ms, sem índice).

### 11.4 Time-Aware RAG — vigência temporal (crítico no Direito, subestimado em geral)

Leis são revogadas, súmulas canceladas — um RAG sem consciência temporal pode devolver jurisprudência superada como se fosse vigente. Solução: função de **decay exponencial** que penaliza documentos antigos:

```
decay(idade) = exp(-ln(2) · max(0, idade_dias - offset) / scale)
score_final = score_relevancia × decay(idade)
```
`scale` = meia-vida em dias (ex. 365 → peso cai à metade em 1 ano); `offset` = período de graça sem penalização. Implementável nativamente no OpenSearch (`function_score` com decay exponencial) ou como re-ranking em Python. Estratégias: decay suave (scale=730d) para jurisprudência histórica; decay agressivo (scale=180d) para normas operacionais; ou filtro rígido (`hard filter`) só documentos vigentes.

### 11.5 DSPy — prompts como programa compilável, não texto artesanal

Prompts escritos à mão são frágeis e não reproduzíveis. DSPy declara **módulos** (ex. `ChainOfThought("context, question -> answer")`), uma **métrica** e um **dataset**; um **otimizador** (`BootstrapFewShot`, `MIPROv2`, `COPRO`) escolhe automaticamente as instruções e os exemplos few-shot que maximizam a métrica — você programa a intenção, o DSPy "compila" o prompt.

### 11.6 Guia de quando usar cada uma

| Técnica | Use quando… | Peso de infra |
|---|---|---|
| Time-Aware | vigência/recência importam (quase sempre no Direito) | leve |
| Compressão | contexto longo, custo/latência alto | médio |
| DSPy | quer prompt robusto e reproduzível, tem dataset | médio |
| ColBERT | precisa de alta precisão de retrieval | pesado (torch/faiss) |
| Multimodal | informação crítica está em imagem/tabela | pesado (CLIP/torch) |

**Onde praticar:** `aula11/scripts/02_time_aware.py`, `03_compressao_llmlingua.py`, `04_colbert_ragatouille.py`, `05_multimodal_clip.py`, `06_dspy_otimizacao.py`; labs `lab1..lab6`.

**Reaproveitamento:** Time-Aware é a técnica mais barata e mais frequentemente esquecida em qualquer domínio onde "atual" importa (não só Direito — preços, políticas, documentação de produto). Implemente-a por padrão sempre que seus documentos tiverem data.

---

## AULA 12 — Projeto Final: Sistema de Ingestão Inteligente + RAG em Produção

Esta aula é a **síntese arquitetural** de todo o curso — não introduz técnica nova, mas mostra como compor as peças das Aulas 2-11 num sistema que **decide sozinho** que técnica de extração, destino de indexação e estratégia de chunking usar, por documento.

### 12.1 Arquitetura da decisão automática

```
Upload → [1] probe (sinais baratos: extensão, presença de texto, imagens)
       → [2] AGENTE (LLM tool-calling, padrão da Aula 10) escolhe a ferramenta de EXTRAÇÃO:
              planilha → pandas | texto → Docling | escaneado/figura → Docling+OCR
       → [3a] HEURÍSTICA escolhe o DESTINO de indexação:
              texto/tabela → OpenSearch | texto longo rico em entidades → LightRAG (grafo, Aula 9)
       → [3b] Se OpenSearch, AVALIADOR heurístico escolhe a TÉCNICA DE CHUNKING
              (fixo | recursivo | sentence-window | semântico | hierárquico | tabela) — Aulas 2 e 6
       → [4] /consulta roteia a busca (OpenSearch ou grafo) e aplica, se OpenSearch,
              a técnica de query enhancement escolhida (baseline | multi_query | rag_fusion | step_back) — Aula 7
```

### 12.2 Decisão de engenharia mais importante desta aula: agente vs. heurística

O projeto usa **agente (LLM)** para a extração (decisão ambígua, depende de "ler" sinais do documento — o LLM brilha aqui) e **heurística transparente** para destino e chunking (decisão auditável baseada em regras claras sobre a estrutura do documento — sem custo de LLM, sem imprevisibilidade). Este é um princípio geral de engenharia de sistemas com LLM: **reserve o LLM para decisões genuinamente ambíguas; use regras determinísticas sempre que a lógica for auditável e estável** — LLM "para tudo" é caro, lento e menos auditável do que necessário.

### 12.3 Stack final integrado

FastAPI (`/ingestao`, `/consulta`, `/health`, `/metrics`) · Haystack (pipeline) · Docling (extração) · Ollama (embeddings) · OpenSearch (busca híbrida + RRF) · LightRAG (grafo) · Groq (LLM com tool calling, `llama-3.3-70b-versatile`) · LangFuse (observabilidade completa, incluindo o grafo via `@observe`). LLM é agnóstico a provedor via `LLM_BASE_URL`/`LLM_API_KEY`/`LLM_MODEL` — o mesmo princípio de portabilidade (trocar só `base_url`) usado desde a Aula 2 (Ollama↔vLLM↔Groq↔OpenAI).

### 12.4 O que copiar diretamente para os seus próprios projetos

- A separação `probe → agente de extração → heurística de destino → heurística de chunking → consulta roteada` é um esqueleto de arquitetura reaproveitável para **qualquer** pipeline de ingestão de documentos heterogêneos, mesmo fora do domínio jurídico.
- O endpoint `/config/prompts` (editar prompts em runtime, persistidos em `prompts.json`, com "restaurar padrão") é um padrão de produção valioso: prompts não devem estar hardcoded no código se você espera iterar sobre eles.
- O relatório de decisão retornado por `/ingestao` (técnica escolhida + motivo) é um padrão de **explicabilidade de pipeline** que vale reaplicar sempre que uma heurística ou agente estiver tomando decisões automáticas.

**Onde estudar:** `aula12/projeto_final/app/` (`extracao.py`, `indexacao.py`, `consulta.py`, `busca_avancada.py`, `grafo.py`, `config.py`); `aula12/projeto_final/README.md` (leitura obrigatória — é o documento de arquitetura mais completo do curso); `aula12/labs/lab1_corpus_e_pipeline.ipynb` até `lab5_avaliacao_ragas.ipynb`.

---

## MATERIAL COMPLEMENTAR — Docling Avançado (aprofundamento da Aula 2)

Este material (`material_docling_extra/teoria/AULA5_TEORIA.md`, 873 linhas) aprofunda tópicos só tocados superficialmente na Aula 2. Vale estudar se seu projeto envolver ingestão pesada de PDFs jurídicos/administrativos reais:

1. Arquitetura interna do Docling em detalhe.
2. **OCR em PDFs escaneados** — parâmetros de qualidade, quando o custo computacional se justifica.
3. **Extração de tabelas estruturadas** — reconstrução linha×coluna, exportação como DataFrame.
4. Estratégias de chunking especificamente pós-Docling.
5. **Pipelines de ingestão em escala** — processamento em lote, paralelismo, cache.
6. Enriquecimento de metadados de chunk (proveniência, seção, página).
7. **Limpeza e normalização de texto jurídico** (ruído de OCR, formatação legada).
8. Limitações e armadilhas conhecidas do Docling.

**Reaproveitamento:** se você for processar um volume grande de PDFs heterogêneos (parte digital, parte escaneada) em produção, este material complementa a Aula 2 com as decisões operacionais (custo de OCR, paralelismo, cache) que a aula principal não cobre em profundidade.

---

## Temas transversais — o que reaparece em quase toda aula

Reconhecer esses padrões recorrentes é o que transforma "decorei 25 técnicas" em "entendo os 5 princípios que geram as 25 técnicas":

1. **Desacoplar unidade de busca da unidade de contexto** — sentence-window (Aula 2) → Parent-Child (Aula 6) → RAPTOR em escala de corpus (Aula 6).
2. **RRF (Reciprocal Rank Fusion)** — fusão BM25+kNN (Aula 4) → fusão de sub-queries no RAG-Fusion (Aula 7). Mesma fórmula, camadas diferentes.
3. **Bi-encoder (rápido, impreciso) vs. cross-encoder (lento, preciso), com ColBERT como meio-termo** — Aula 3 (reranking) e Aula 11 (ColBERT).
4. **LLM-as-Judge com roteamento condicional** — avaliador do CRAG (Aula 8) → classificador de complexidade do Adaptive RAG (Aula 10). Mesmo padrão, granularidades diferentes.
5. **Reservar o LLM para decisões ambíguas; usar heurística/regra para decisões auditáveis** — explícito na Aula 12, implícito em todo o curso.
6. **Nunca confiar em melhoria não medida** — todo o aparato de avaliação (Aula 5) existe para validar (ou refutar) cada técnica das Aulas 6-11.
7. **Soberania de dados / execução local** — Ollama (Aula 1-2), vLLM (Aula 4+), OpenSearch on-premise (Aula 9) — recorrente por causa do domínio jurídico, mas o padrão de arquitetura "tudo trocável via `base_url`" é genericamente valioso.

---

## Checklist de domínio — perguntas para autoavaliação por aula

Cada `teoria/AULAx_TEORIA.md` já traz "Perguntas para Reflexão" com respostas — use-as como teste real de compreensão, não apenas leitura passiva. Antes de avançar de aula, você deve conseguir responder sem consultar o material:

- **Aula 1**: por que um embedding estático falha para "banco" em dois contextos, e por que isso não acontece com BERT?
- **Aula 2**: por que usar modelos de embedding diferentes na indexação e na query quebra silenciosamente o retrieval?
- **Aula 3**: por que um cross-encoder não pode substituir um bi-encoder na primeira etapa de busca em 1M documentos?
- **Aula 4**: por que RRF é preferível a min-max normalization para fundir BM25 e kNN?
- **Aula 5**: dado Faithfulness=0.65 e Context Recall=0.95, qual componente do pipeline você investiga primeiro?
- **Aula 6**: por que RAPTOR falha com corpus de 5 documentos?
- **Aula 7**: por que Step-Back pode piorar o recall para uma query já abstrata?
- **Aula 8**: por que CRAG é mais fácil de adotar em produção do que Self-RAG?
- **Aula 9**: dê um exemplo de pergunta que só o modo "global" do LightRAG resolveria bem.
- **Aula 10**: por que a descrição de uma tool importa mais do que o nome dela?
- **Aula 11**: em que cenário ColBERT vale o custo de índice 3-5x maior?
- **Aula 12**: por que o projeto final usa agente para extração mas heurística para chunking?

---

## Roteiro para seu próprio projeto (aplicando o curso fora do domínio jurídico)

1. **Baseline (Aulas 1-2)**: Naive RAG com chunking recursivo + BGE-M3 + OpenSearch denso. Meça com RAGAS antes de qualquer otimização.
2. **Retrieval de alto ROI (Aulas 3-4)**: busca híbrida (BM25+kNN) com RRF + reranking cross-encoder nos top-100→top-5. Isso sozinho costuma resolver a maior parte dos problemas de qualidade.
3. **Avaliação contínua (Aula 5)**: RAGAS no CI, LangFuse em produção, com metas explícitas por métrica.
4. **Escolha cirúrgica das Aulas 6-11**: não implemente tudo. Pergunte: meu corpus tem estrutura hierárquica clara? (Parent-Child) Preciso de síntese multi-documento? (RAPTOR) Meus usuários usam vocabulário informal? (HyDE/Multi-Query) Minha pergunta típica é multi-hop entre entidades? (Graph RAG) Meu domínio tem vigência temporal? (Time-Aware) Tenho contexto longo demais? (Compressão) Preciso de automação de decisão multi-etapa? (Agentic/Adaptive RAG).
5. **Arquitetura de produção (Aula 12)**: agente só onde a decisão é ambígua; heurística/regra em todo o resto; observabilidade (LangFuse) e explicabilidade (relatório de decisão) desde o primeiro deploy.

---

*Documento gerado a partir da leitura integral dos materiais de teoria das Aulas 1–11, do README do Projeto Final (Aula 12) e do índice do material complementar de Docling, no repositório `IBMEC-MBA-RAG`.*

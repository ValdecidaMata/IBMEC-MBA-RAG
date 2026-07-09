"""Monta avaliacao/dataset.json final: documentos (por artigo) + queries_benchmark.

As perguntas foram escritas manualmente, em linguagem coloquial (sem copiar os
termos exatos do texto legal), e verificadas contra o conteudo real dos artigos
extraidos por PyMuPDF (ver documentos_artigos_pymupdf.json).
"""
import json
from pathlib import Path

PASTA = Path(__file__).resolve().parent

QUERIES = [
    # --- Factuais (1 documento) ---
    {
        "id": "Q01",
        "tipo": "factual",
        "query": "Quem pode chamar um plebiscito ou referendo no Brasil?",
        "relevancia": {"Art14": 2},
        "resposta_referencia": "O povo exerce a soberania popular pelo sufrágio universal e voto direto e secreto, inclusive por meio de plebiscito, referendo e iniciativa popular (Art. 14).",
    },
    {
        "id": "Q02",
        "tipo": "factual",
        "query": "Quais entes formam a organização político-administrativa do país?",
        "relevancia": {"Art18": 2},
        "resposta_referencia": "A União, os Estados, o Distrito Federal e os Municípios, todos autônomos (Art. 18).",
    },
    {
        "id": "Q03",
        "tipo": "factual",
        "query": "Quais princípios a administração pública precisa seguir?",
        "relevancia": {"Art37": 2},
        "resposta_referencia": "Legalidade, impessoalidade, moralidade, publicidade e eficiência (Art. 37).",
    },
    {
        "id": "Q04",
        "tipo": "factual",
        "query": "O que é o Ministério Público e quais ramos ele tem?",
        "relevancia": {"Art128": 2},
        "resposta_referencia": "Abrange o Ministério Público da União (Federal, do Trabalho, Militar, do Distrito Federal e Territórios) e os Ministérios Públicos dos Estados (Art. 128).",
    },
    {
        "id": "Q05",
        "tipo": "factual",
        "query": "Quais impostos e taxas os governos podem cobrar da população?",
        "relevancia": {"Art145": 2},
        "resposta_referencia": "Impostos, taxas (pelo poder de polícia ou por serviços públicos) e contribuições de melhoria (Art. 145).",
    },
    {
        "id": "Q06",
        "tipo": "factual",
        "query": "Como funciona a aposentadoria de quem trabalha no serviço público?",
        "relevancia": {"Art40": 2},
        "resposta_referencia": "Regime próprio de previdência social, de caráter contributivo e solidário, com contribuição do ente federativo, dos servidores ativos, aposentados e pensionistas (Art. 40).",
    },
    {
        "id": "Q07",
        "tipo": "factual",
        "query": "O que o Supremo Tribunal Federal tem o poder de julgar?",
        "relevancia": {"Art102": 2},
        "resposta_referencia": "Compete ao STF, precipuamente, a guarda da Constituição, incluindo julgar ações diretas de inconstitucionalidade, entre outras competências originárias (Art. 102).",
    },
    {
        "id": "Q08",
        "tipo": "factual",
        "query": "O que a lei diz sobre a proteção da família e do casamento?",
        "relevancia": {"Art226": 2},
        "resposta_referencia": "A família é a base da sociedade e tem especial proteção do Estado; o casamento é civil e sua celebração é gratuita (Art. 226).",
    },
    {
        "id": "Q09",
        "tipo": "factual",
        "query": "Quais tribunais e órgãos fazem parte do Poder Judiciário?",
        "relevancia": {"Art92": 2},
        "resposta_referencia": "STF, CNJ, STJ, Tribunais Regionais Federais e juízes federais, tribunais e juízes do trabalho, eleitorais, militares e dos Estados/DF (Art. 92).",
    },
    # --- Reformuláveis (vocabulário coloquial, boas para multi_query/step_back/HyDE) ---
    {
        "id": "Q10",
        "tipo": "reformulavel",
        "query": "Criança e adolescente têm prioridade em algum direito perante a lei?",
        "relevancia": {"Art227": 2},
        "resposta_referencia": "É dever da família, da sociedade e do Estado assegurar à criança, ao adolescente e ao jovem, com absoluta prioridade, direitos como vida, saúde, alimentação e educação (Art. 227).",
    },
    {
        "id": "Q11",
        "tipo": "reformulavel",
        "query": "Existe alguma regra sobre como a economia do país deve funcionar?",
        "relevancia": {"Art170": 2},
        "resposta_referencia": "A ordem econômica é fundada na valorização do trabalho humano e na livre iniciativa, com fim de assegurar existência digna conforme a justiça social (Art. 170).",
    },
    {
        "id": "Q12",
        "tipo": "reformulavel",
        "query": "Como funciona a aposentadoria de quem trabalha na iniciativa privada?",
        "relevancia": {"Art201": 2},
        "resposta_referencia": "Pelo Regime Geral de Previdência Social, de caráter contributivo e filiação obrigatória, observado o equilíbrio financeiro e atuarial (Art. 201).",
    },
    {
        "id": "Q13",
        "tipo": "reformulavel",
        "query": "Quem tem o poder de sugerir uma lei nova no país?",
        "relevancia": {"Art61": 2},
        "resposta_referencia": "Qualquer membro ou Comissão da Câmara, do Senado ou do Congresso, o Presidente da República, o STF, os Tribunais Superiores, o Procurador-Geral da República e os cidadãos, nos casos previstos (Art. 61).",
    },
    {
        "id": "Q14",
        "tipo": "reformulavel",
        "query": "O que está escrito sobre a preservação da natureza no país?",
        "relevancia": {"Art225": 2},
        "resposta_referencia": "Todos têm direito ao meio ambiente ecologicamente equilibrado, bem de uso comum do povo, e o poder público e a coletividade têm o dever de defendê-lo e preservá-lo (Art. 225).",
    },
    {
        "id": "Q15",
        "tipo": "reformulavel",
        "query": "Tem algum caso em que o governo não pode cobrar imposto?",
        "relevancia": {"Art150": 2},
        "resposta_referencia": "A Constituição veda à União, Estados, DF e Municípios cobrar tributos em diversas hipóteses de imunidade e limitação ao poder de tributar (Art. 150).",
    },
    # --- Temáticas / multi-hop (mais de um documento) ---
    {
        "id": "Q16",
        "tipo": "multi_hop",
        "query": "Quais direitos sociais os trabalhadores têm e quem garante saúde para todo mundo?",
        "relevancia": {"Art6": 2, "Art196": 2, "Art7": 1},
        "resposta_referencia": "São direitos sociais educação, saúde, alimentação, trabalho, moradia, entre outros (Art. 6); a saúde é direito de todos e dever do Estado (Art. 196); e os trabalhadores urbanos e rurais têm direitos específicos (Art. 7).",
    },
    {
        "id": "Q17",
        "tipo": "multi_hop",
        "query": "Quem é responsável por cuidar da saúde pública e proteger o meio ambiente ao mesmo tempo?",
        "relevancia": {"Art23": 2, "Art196": 1, "Art225": 1},
        "resposta_referencia": "É competência comum da União, Estados, DF e Municípios cuidar da saúde e proteger o meio ambiente (Art. 23), complementada pelo dever do Estado com a saúde (Art. 196) e a preservação ambiental (Art. 225).",
    },
    {
        "id": "Q18",
        "tipo": "multi_hop",
        "query": "Como o Brasil garante o acesso à escola e quem tem esse direito garantido de forma mais ampla?",
        "relevancia": {"Art205": 2, "Art6": 1},
        "resposta_referencia": "A educação é direito de todos e dever do Estado e da família (Art. 205), sendo também um dos direitos sociais listados de forma geral (Art. 6).",
    },
    {
        "id": "Q19",
        "tipo": "multi_hop",
        "query": "Como os três poderes se organizam e quais tribunais existem para julgar as pessoas?",
        "relevancia": {"Art2": 2, "Art92": 2, "Art44": 1},
        "resposta_referencia": "São Poderes da União o Legislativo, o Executivo e o Judiciário, independentes e harmônicos entre si (Art. 2); o Judiciário é composto por STF, CNJ, STJ e outros tribunais (Art. 92); o Legislativo é exercido pelo Congresso Nacional (Art. 44).",
    },
    {
        "id": "Q20",
        "tipo": "multi_hop",
        "query": "Dá para mudar a Constituição? Se sim, quem propõe e qual a diferença para propor uma lei comum?",
        "relevancia": {"Art60": 2, "Art61": 1, "Art59": 1},
        "resposta_referencia": "A Constituição pode ser emendada por proposta de um terço da Câmara ou Senado, do Presidente ou de mais da metade das Assembleias Legislativas (Art. 60), diferente da iniciativa de leis ordinárias (Art. 61), dentro do processo legislativo geral (Art. 59).",
    },
    {
        "id": "Q21",
        "tipo": "multi_hop",
        "query": "Quais garantias uma pessoa tem para proteger sua vida e liberdade, e isso também vale de forma especial para crianças?",
        "relevancia": {"Art5": 2, "Art227": 1},
        "resposta_referencia": "O Art. 5 garante a todos a inviolabilidade do direito à vida, liberdade, igualdade, segurança e propriedade; o Art. 227 estende proteção especial e prioritária a crianças, adolescentes e jovens.",
    },
    {
        "id": "Q22",
        "tipo": "multi_hop",
        "query": "Quem pode cobrar tributos e quais são os limites para essa cobrança?",
        "relevancia": {"Art145": 2, "Art150": 2, "Art23": 1},
        "resposta_referencia": "União, Estados, DF e Municípios podem instituir impostos, taxas e contribuições de melhoria (Art. 145), respeitando limitações e imunidades tributárias (Art. 150), e compartilhando competências comuns entre si (Art. 23).",
    },
]


def main():
    documentos = json.loads((PASTA / "documentos_artigos_pymupdf.json").read_text(encoding="utf-8"))
    dataset = {"documentos": documentos, "queries_benchmark": QUERIES}
    (PASTA / "dataset.json").write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    tipos = {}
    for q in QUERIES:
        tipos[q["tipo"]] = tipos.get(q["tipo"], 0) + 1
    print(f"dataset.json salvo: {len(documentos)} documentos, {len(QUERIES)} perguntas -> {tipos}")


if __name__ == "__main__":
    main()

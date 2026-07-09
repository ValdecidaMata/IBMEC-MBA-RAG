"""
Constroi a base de artigos (gabarito) a partir do texto da Constituicao.

Uso:
    python construir_dataset.py --fonte docling|pymupdf

Faz o split do texto principal da Constituicao (exclui ADCT, que reinicia a
numeracao de artigos) em um "documento" por artigo (Art. N, incluindo sufixos
de emenda como Art. 149-A). O resultado alimenta avaliacao/dataset.json (campo
"documentos"), que e completado manualmente com as perguntas em
avaliacao/perguntas.py.
"""
import argparse
import json
import re
import sys
from pathlib import Path

PASTA = Path(__file__).resolve().parent
RAIZ_PROJETO_FINAL = PASTA.parent

PADRAO_ARTIGO = re.compile(r"^.{0,3}Art\.\s*(\d+)[ºo°]?(-[A-Z])?\.?\s", re.MULTILINE)


def extrair_texto_pymupdf(caminho_pdf: str) -> str:
    import fitz

    doc = fitz.open(caminho_pdf)
    return "".join(p.get_text() for p in doc)


def extrair_texto_docling(caminho_pdf: str) -> str:
    sys.path.insert(0, str(RAIZ_PROJETO_FINAL))
    from app import extracao

    _, _, _, _, dados = extracao.decidir_e_extrair(caminho_pdf)
    return dados["conteudo"]


def texto_principal(txt: str) -> str:
    """Recorta so o corpo principal da Constituicao (exclui ADCT)."""
    fim = txt.find("DISPOSIÇÕES CONSTITUCIONAIS TRANSITÓRIAS")
    if fim == -1:
        fim = len(txt)
    inicio = txt.find("TÍTULO I")
    if inicio == -1:
        inicio = 0
    return txt[inicio:fim]


def dividir_por_artigo(txt_principal: str) -> list[dict]:
    matches = list(PADRAO_ARTIGO.finditer(txt_principal))
    documentos = []
    for i, m in enumerate(matches):
        numero, sufixo = m.group(1), (m.group(2) or "").replace("-", "")
        doc_id = f"Art{numero}{sufixo}"
        inicio = m.end()
        fim = matches[i + 1].start() if i + 1 < len(matches) else len(txt_principal)
        corpo = txt_principal[inicio:fim].strip()
        documentos.append({"id": doc_id, "artigo": f"Art. {numero}{('-' + sufixo) if sufixo else ''}", "texto": corpo})
    return documentos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonte", choices=["pymupdf", "docling"], default="pymupdf")
    ap.add_argument("--pdf", default=str(RAIZ_PROJETO_FINAL / "uploads" / "Constituicao-Compilado.pdf"))
    ap.add_argument("--saida", default=str(PASTA / "documentos_artigos.json"))
    args = ap.parse_args()

    txt = extrair_texto_docling(args.pdf) if args.fonte == "docling" else extrair_texto_pymupdf(args.pdf)
    principal = texto_principal(txt)
    documentos = dividir_por_artigo(principal)

    ids_numericos = []
    for d in documentos:
        base = re.match(r"Art(\d+)", d["id"])
        if base:
            ids_numericos.append(int(base.group(1)))
    faltantes = sorted(set(range(1, 251)) - set(ids_numericos))

    print(f"fonte={args.fonte} n_chars_total={len(txt)} n_chars_principal={len(principal)}")
    print(f"n_artigos_encontrados={len(documentos)} n_numeros_unicos={len(set(ids_numericos))}")
    print(f"numeros de artigo ausentes (1-250): {faltantes}")

    Path(args.saida).write_text(json.dumps(documentos, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"salvo em {args.saida}")


if __name__ == "__main__":
    main()

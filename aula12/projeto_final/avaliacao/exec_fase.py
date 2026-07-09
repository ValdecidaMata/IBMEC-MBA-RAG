"""Roda uma tecnica contra o dataset e imprime/grava as metricas de recuperacao.
Uso: python exec_fase.py <exp> <fase> "<mudanca>" <tecnica> <top_k> "<observacao>"
"""
import sys
import comum

exp, fase, mudanca, tecnica, top_k, obs = sys.argv[1:7]
top_k = int(top_k)
dataset = comum.carregar_dataset()
m = comum.avaliar_tecnica(tecnica, top_k, dataset)
linha = {"exp": exp, "fase": fase, "mudanca": mudanca, **m, "observacao": obs}
print(linha)
comum.gravar_linha_csv("resultados.csv", linha)

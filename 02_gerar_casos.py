"""Passo 2: aplica as MRs e gera os casos subsequentes (follow-up).

Saídas:
    dados/casos_followup.csv  - todos os casos gerados
    dados/revisao_manual.xlsx - casos para você revisar (MRs geradas por LLM e a MR3,
        cuja remoção de acentos pode mudar o sentido: é/e, está/esta).
        Abra no Excel ou no Google Planilhas e preencha a coluna "aprovado" com S ou N.
        Para corrigir em vez de descartar, escreva a versão certa em "correcao_premissa" /
        "correcao_hipotese" e marque S. Em "motivo", anote por que descartou ou corrigiu.
        O passo 3 só usa esses casos se aprovados.
A geração é retomável: cada reescrita do LLM é guardada em dados/cache_gerador.jsonl.
Se a cota diária gratuita acabar, rode de novo no dia seguinte; o que já foi gerado é reaproveitado.
"""
import json
import sys

import pandas as pd

import modelos
from config import DADOS, MODELO_GERADOR, MRS_ATIVAS
from transformacoes import MRS, aplicar_mr

destino = DADOS / "revisao_manual.xlsx"
if destino.exists():
    sys.exit(f"{destino} já existe e pode conter sua revisão. Renomeie ou apague o arquivo antes de gerar de novo.")

fonte = pd.read_csv(DADOS / "casos_fonte.csv")
ATIVAS = {c: MRS[c] for c in MRS_ATIVAS}
precisa_llm = any(m["metodo"] == "llm" for m in ATIVAS.values())

CACHE = DADOS / "cache_gerador.jsonl"
cache = {}
if CACHE.exists():
    for linha in CACHE.read_text(encoding="utf-8").splitlines():
        r = json.loads(linha)
        cache[r["prompt"]] = r["resposta"]
_gerar_llm = None


def gerar(prompt):
    """Chama o gerador só para prompts ainda não respondidos e guarda a resposta."""
    global _gerar_llm
    if prompt in cache:
        return cache[prompt]
    if _gerar_llm is None:
        _gerar_llm = modelos.carregar(MODELO_GERADOR)
    resposta = _gerar_llm(prompt)
    cache[prompt] = resposta
    with open(CACHE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"prompt": prompt, "resposta": resposta}, ensure_ascii=False) + "\n")
    return resposta


linhas = []
for _, caso in fonte.iterrows():
    for codigo, mr in ATIVAS.items():
        try:
            resultado = aplicar_mr(codigo, caso.premissa, caso.hipotese, caso.rotulo, gerar)
        except modelos.LimiteDiario:
            sys.exit(f"\nA cota diária gratuita do gerador acabou em {caso.id_fonte}. "
                     f"{len(cache)} reescritas já estão salvas. Rode esta etapa de novo amanhã: ela continua de onde parou.")
        if resultado is None:
            continue
        p, h = resultado
        linhas.append({
            "id_caso": f"{caso.id_fonte}-{codigo}", "id_fonte": caso.id_fonte, "mr": codigo,
            "metodo": mr["metodo"], "premissa_fonte": caso.premissa, "hipotese_fonte": caso.hipotese,
            "premissa": p, "hipotese": h, "rotulo_fonte": caso.rotulo,
        })
    print(f"{caso.id_fonte} ok", end="\r")

casos = pd.DataFrame(linhas)
casos.to_csv(DADOS / "casos_followup.csv", index=False)

revisao = casos[casos.mr.map(lambda c: MRS[c]["revisar"])].copy()
revisao = revisao[["id_caso", "mr", "premissa_fonte", "hipotese_fonte", "premissa", "hipotese"]]
revisao.insert(1, "nome_mr", revisao.mr.map(lambda c: MRS[c]["nome"]))
for col in ("aprovado", "correcao_premissa", "correcao_hipotese", "motivo"):
    revisao[col] = ""
revisao.to_excel(destino, index=False)

# formatação para facilitar a leitura
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation
wb = load_workbook(destino); ws = wb.active
larguras = {"A": 12, "B": 18, "C": 6, "D": 38, "E": 32, "F": 38, "G": 32, "H": 10, "I": 30, "J": 30, "K": 30}
for col, w in larguras.items():
    ws.column_dimensions[col].width = w
for linha in ws.iter_rows(min_row=2):
    for cel in linha:
        cel.alignment = Alignment(wrap_text=True, vertical="top")
amarelo = PatternFill("solid", fgColor="FFF2CC")
for cel in ws[1]:
    cel.font = Font(bold=True)
for col in "HIJK":
    for cel in ws[col][1:]:
        cel.fill = amarelo
dv = DataValidation(type="list", formula1='"S,N"', allow_blank=True)
ws.add_data_validation(dv); dv.add(f"H2:H{ws.max_row}")
ws.freeze_panes = "D2"
wb.save(destino)

print(f"\n{len(casos)} casos follow-up gerados")
print(casos.groupby("mr").size().rename("casos"))
print(f"Revise {len(revisao)} casos em {destino} antes do passo 3.")

"""Passo 5 (opcional): compara a instrução original (p1) com a alternativa (p2).

Uso:
    python 05_comparar_instrucoes.py

Funciona mesmo com execuções parciais: depois de rodar só os casos fonte
(03_executar.py --prompt p2 --so-fonte <modelo>), já mostra a proporção de respostas SIM,
que é o que decide se vale a pena fazer a execução completa com a nova instrução.

Saída: resultados/tabela_sensibilidade_instrucao.csv
"""
import json
from collections import Counter

import numpy as np
import pandas as pd

from config import DADOS, PROMPTS, RESULTADOS, arquivo_respostas
from transformacoes import INVERSAO, MRS


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    centro = (p + z * z / (2 * n)) / (1 + z * z / n)
    margem = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centro - margem), min(1.0, centro + margem))


def majoritaria(respostas):
    c = Counter(respostas).most_common()
    if len(c) > 1 and c[0][1] == c[1][1]:
        return "INVALIDA"
    return c[0][0]


fonte = pd.read_csv(DADOS / "casos_fonte.csv")
gabarito = dict(zip(fonte.id_fonte, fonte.rotulo.map({"ENTAILMENT": "SIM", "NONE": "NAO"})))

linhas = []
for prompt in PROMPTS:
    arq = arquivo_respostas(prompt)
    if not arq.exists():
        continue
    brutas = pd.DataFrame([json.loads(l) for l in open(arq, encoding="utf-8")])
    maj = brutas.groupby(["modelo", "id_caso"]).resposta.agg(majoritaria)
    for modelo in sorted(brutas.modelo.unique()):
        m = maj[modelo]
        fontes = m[m.index.isin(gabarito)]
        ouro = fontes.index.map(gabarito)
        linha = {
            "instrucao": prompt, "modelo": modelo, "casos_fonte": len(fontes),
            "respostas_SIM_%": 100 * (fontes == "SIM").mean(),
            "acuracia_%": 100 * (fontes.values == ouro.values).mean(),
            "acerto_com_implicacao_%": 100 * (fontes[ouro == "SIM"] == "SIM").mean(),
            "acerto_sem_implicacao_%": 100 * (fontes[ouro == "NAO"] == "NAO").mean(),
        }
        # violações das relações de preservação, separadas pela resposta ao caso fonte
        pares = []
        for id_caso, r in m.items():
            if "-" not in id_caso:
                continue
            id_fonte, mr = id_caso.split("-")
            if MRS[mr]["tipo"] == INVERSAO or id_fonte not in m.index:
                continue
            rf = m[id_fonte]
            if "INVALIDA" in (rf, r):
                continue
            pares.append((rf, r != rf))
        for rf in ("SIM", "NAO"):
            sel = [v for f, v in pares if f == rf]
            k, n = sum(sel), len(sel)
            lo, hi = wilson(k, n)
            linha[f"pares_fonte_{rf}"] = n
            linha[f"violacao_fonte_{rf}_%"] = 100 * k / n if n else np.nan
            linha[f"ic95_fonte_{rf}"] = f"{100 * lo:.1f}–{100 * hi:.1f}" if n else ""
        k, n = sum(v for _, v in pares), len(pares)
        linha["pares_total"], linha["violacao_total_%"] = n, (100 * k / n if n else np.nan)
        linhas.append(linha)

tabela = pd.DataFrame(linhas).round(1)
tabela.to_csv(RESULTADOS / "tabela_sensibilidade_instrucao.csv", index=False)
pd.set_option("display.width", 200)
print(tabela[["instrucao", "modelo", "casos_fonte", "respostas_SIM_%", "acuracia_%",
              "acerto_com_implicacao_%", "pares_fonte_SIM", "violacao_fonte_SIM_%", "violacao_total_%"]].to_string(index=False))

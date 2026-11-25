"""Passo 4: calcula as métricas e gera tabelas e gráficos para o Capítulo 4.

Saídas em resultados/:
    tabela_revisao.csv       - casos gerados, aprovados, corrigidos e descartados por MR (Tabela 1)
    tabela_violacoes.csv     - taxa de violação por MR x modelo, com IC de 95% (Wilson)
    tabela_modelos.csv       - acurácia nos casos fonte, não determinismo e respostas inválidas
    violacoes_detalhadas.csv - cada violação com os textos, para a análise qualitativa
    testes_estatisticos.csv  - qui-quadrado/Fisher comparando os modelos em cada MR
    tabela_vies.csv          - proporção de respostas SIM, acerto por classe e linha de base "sempre NÃO"
    tabela_violacoes_por_resposta_fonte.csv - taxa de violação (MRs de preservação) separada
                               pela resposta do modelo no caso fonte (SIM x NÃO), com IC de 95%
    testes_por_resposta_fonte.csv - Fisher entre pares de modelos, dentro de cada resposta fonte
    grafico_violacoes.png    - gráfico de barras para o texto
    grafico_por_resposta_fonte.png - taxa de violação por resposta fonte, com IC de 95%
"""
import json
from collections import Counter

import numpy as np
import pandas as pd

import sys

from config import DADOS, RESULTADOS, arquivo_respostas

# python 04_analisar.py            -> instrução original; saídas em resultados/
# python 04_analisar.py --prompt p2 -> instrução p2; saídas em resultados/p2/
PROMPT = sys.argv[sys.argv.index("--prompt") + 1] if "--prompt" in sys.argv else "p1"
SAIDA = RESULTADOS if PROMPT == "p1" else RESULTADOS / PROMPT
SAIDA.mkdir(exist_ok=True)
from transformacoes import INVERSAO, MRS


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    centro = (p + z * z / (2 * n)) / (1 + z * z / n)
    margem = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centro - margem), min(1.0, centro + margem))


def resposta_majoritaria(respostas):
    """Voto majoritário entre as repetições; empate ou maioria inválida -> INVALIDA."""
    contagem = Counter(respostas).most_common()
    if len(contagem) > 1 and contagem[0][1] == contagem[1][1]:
        return "INVALIDA"
    return contagem[0][0]


def carregar():
    with open(arquivo_respostas(PROMPT), encoding="utf-8") as f:
        brutas = pd.DataFrame([json.loads(l) for l in f])
    agregadas = (
        brutas.groupby(["modelo", "id_caso"])["resposta"]
        .agg(maj=resposta_majoritaria, estavel=lambda s: s.nunique() == 1, n="size")
        .reset_index()
    )
    return brutas, agregadas


def main():
    brutas, ag = carregar()
    fonte = pd.read_csv(DADOS / "casos_fonte.csv")
    followup = pd.read_csv(DADOS / "casos_followup.csv")
    gabarito = dict(zip(fonte.id_fonte, fonte.rotulo.map({"ENTAILMENT": "SIM", "NONE": "NAO"})))

    # ---- Tabela por modelo: acurácia, não determinismo, inválidas
    linhas_modelo = []
    for modelo, g in ag.groupby("modelo"):
        gf = g[g.id_caso.isin(gabarito)]
        acertos = (gf.maj == gf.id_caso.map(gabarito)).sum()
        linhas_modelo.append({
            "modelo": modelo,
            "acuracia_fonte": acertos / len(gf),
            "entradas_instaveis_%": 100 * (~g.estavel).mean(),
            "respostas_invalidas_%": 100 * (brutas[brutas.modelo == modelo].resposta == "INVALIDA").mean(),
            "entradas": len(g),
            "inicio_execucao": brutas[brutas.modelo == modelo].horario.min()[:10],
            "fim_execucao": brutas[brutas.modelo == modelo].horario.max()[:10],
        })
    pd.DataFrame(linhas_modelo).round(3).to_csv(SAIDA / "tabela_modelos.csv", index=False)

    # ---- Tabela 1: resultado da revisão manual por MR
    rev_path = DADOS / "revisao_manual.xlsx"
    rev = pd.read_excel(rev_path, dtype=str).fillna("") if rev_path.exists() else pd.DataFrame(columns=["id_caso", "mr", "aprovado", "correcao_premissa", "correcao_hipotese"])
    linhas_rev = []
    for mr in sorted(followup.mr.unique()):
        g = rev[rev.mr == mr]
        aprov = g.aprovado.str.upper().str.strip() == "S"
        corrig = aprov & ((g.correcao_premissa.str.strip() != "") | (g.correcao_hipotese.str.strip() != ""))
        gerados = (followup.mr == mr).sum()
        revisada = MRS[mr]["revisar"]
        linhas_rev.append({"mr": mr, "nome_mr": MRS[mr]["nome"], "gerados": gerados,
                           "aprovados_sem_alteracao": int((aprov & ~corrig).sum()) if revisada else gerados,
                           "corrigidos": int(corrig.sum()) if revisada else 0,
                           "descartados": int((~aprov).sum()) if revisada else 0,
                           "utilizados": int(aprov.sum()) if revisada else gerados,
                           "revisao_manual": "sim" if revisada else "não (regra)"})
    tab_rev = pd.DataFrame(linhas_rev)
    tab_rev.loc[len(tab_rev)] = {"mr": "Total", "nome_mr": "", "gerados": tab_rev.gerados.sum(),
                                 "aprovados_sem_alteracao": tab_rev.aprovados_sem_alteracao.sum(), "corrigidos": tab_rev.corrigidos.sum(),
                                 "descartados": tab_rev.descartados.sum(), "utilizados": tab_rev.utilizados.sum(), "revisao_manual": ""}
    tab_rev.to_csv(SAIDA / "tabela_revisao.csv", index=False)

    # os modelos receberam o texto corrigido na revisão manual (ver 03_executar.py);
    # aplica as mesmas correções aqui para que as tabelas mostrem o texto realmente usado
    for linha in rev.itertuples():
        for campo in ("premissa", "hipotese"):
            novo = str(getattr(linha, f"correcao_{campo}")).strip()
            if novo:
                followup.loc[followup.id_caso == linha.id_caso, campo] = novo

    # ---- Verificação das relações metamórficas
    maj = ag.set_index(["modelo", "id_caso"])["maj"]
    registros = []
    for modelo in ag.modelo.unique():
        for c in followup.itertuples():
            if (modelo, c.id_caso) not in maj.index:
                continue  # caso reprovado na revisão manual ou ainda não executado
            r_fonte, r_follow = maj[(modelo, c.id_fonte)], maj[(modelo, c.id_caso)]
            tipo = MRS[c.mr]["tipo"]
            if "INVALIDA" in (r_fonte, r_follow):
                situacao = "indeterminado"
            elif tipo == INVERSAO:
                if r_fonte != "SIM":
                    situacao = "nao_aplicavel"  # a MR6 só tem expectativa definida se a fonte foi SIM
                else:
                    situacao = "violacao" if r_follow != "NAO" else "satisfeita"
            else:
                situacao = "violacao" if r_follow != r_fonte else "satisfeita"
            registros.append({
                "modelo": modelo, "mr": c.mr, "id_caso": c.id_caso, "situacao": situacao,
                "resp_fonte": r_fonte, "resp_followup": r_follow,
                "premissa_fonte": c.premissa_fonte, "hipotese_fonte": c.hipotese_fonte,
                "premissa_followup": c.premissa, "hipotese_followup": c.hipotese,
            })
    reg = pd.DataFrame(registros)

    linhas = []
    for (modelo, mr), g in reg.groupby(["modelo", "mr"]):
        avaliados = g[g.situacao.isin(["violacao", "satisfeita"])]
        k, n = (avaliados.situacao == "violacao").sum(), len(avaliados)
        lo, hi = wilson(k, n)
        linhas.append({
            "modelo": modelo, "mr": mr, "nome_mr": MRS[mr]["nome"], "casos_avaliados": n,
            "violacoes": k, "taxa_violacao_%": 100 * k / n if n else np.nan,
            "ic95_inf_%": 100 * lo, "ic95_sup_%": 100 * hi,
            "indeterminados": (g.situacao == "indeterminado").sum(),
        })
    tabela = pd.DataFrame(linhas).round(2)
    tabela.to_csv(SAIDA / "tabela_violacoes.csv", index=False)
    reg[reg.situacao == "violacao"].to_csv(SAIDA / "violacoes_detalhadas.csv", index=False)

    # ---- Teste estatístico: os modelos diferem em cada MR?
    try:
        from scipy.stats import chi2_contingency, fisher_exact
        testes = []
        for mr, g in tabela.groupby("mr"):
            cont = np.array([[r.violacoes, r.casos_avaliados - r.violacoes] for r in g.itertuples()])
            if (cont.sum(axis=0) == 0).any() or len(cont) < 2:
                continue
            if len(cont) == 2:
                _, p = fisher_exact(cont)
                teste = "Fisher"
            else:
                _, p, _, _ = chi2_contingency(cont)
                teste = "Qui-quadrado"
            testes.append({"mr": mr, "teste": teste, "p_valor": round(p, 4)})
        pd.DataFrame(testes).to_csv(SAIDA / "testes_estatisticos.csv", index=False)
    except ImportError:
        print("scipy não instalado: testes estatísticos ignorados")

    # ---- Viés de resposta: um modelo que responde quase sempre NÃO quase nunca viola as MRs
    # de preservação. Por isso a taxa de violação também é apresentada separada pela resposta
    # que o modelo deu no caso fonte.
    linhas_vies = []
    for modelo, g in ag[ag.id_caso.isin(gabarito)].groupby("modelo"):
        ouro = g.id_caso.map(gabarito)
        linhas_vies.append({
            "modelo": modelo,
            "respostas_SIM_%": 100 * (g.maj == "SIM").mean(),
            "acuracia_%": 100 * (g.maj == ouro).mean(),
            "acerto_com_implicacao_%": 100 * (g[ouro == "SIM"].maj == "SIM").mean(),
            "acerto_sem_implicacao_%": 100 * (g[ouro == "NAO"].maj == "NAO").mean(),
        })
    linhas_vies.append({"modelo": "linha de base: sempre NÃO", "respostas_SIM_%": 0.0, "acuracia_%": 50.0,
                        "acerto_com_implicacao_%": 0.0, "acerto_sem_implicacao_%": 100.0})
    pd.DataFrame(linhas_vies).round(1).to_csv(SAIDA / "tabela_vies.csv", index=False)

    pres = reg[reg.mr.map(lambda c: MRS[c]["tipo"]) != INVERSAO]
    pres = pres[pres.situacao.isin(["violacao", "satisfeita"])].copy()
    pres["viol"] = pres.situacao == "violacao"
    linhas_rf = []
    for (modelo, rf), g in pres.groupby(["modelo", "resp_fonte"]):
        k, n = int(g.viol.sum()), len(g)
        lo, hi = wilson(k, n)
        linhas_rf.append({"modelo": modelo, "resposta_fonte": rf, "casos_avaliados": n, "violacoes": k,
                          "taxa_violacao_%": 100 * k / n, "ic95_inf_%": 100 * lo, "ic95_sup_%": 100 * hi})
    pd.DataFrame(linhas_rf).round(2).to_csv(SAIDA / "tabela_violacoes_por_resposta_fonte.csv", index=False)
    try:
        from itertools import combinations
        from scipy.stats import fisher_exact
        testes_rf = []
        for rf, g in pres.groupby("resp_fonte"):
            for a, b in combinations(sorted(g.modelo.unique()), 2):
                va, vb = g[g.modelo == a].viol, g[g.modelo == b].viol
                _, p = fisher_exact([[va.sum(), len(va) - va.sum()], [vb.sum(), len(vb) - vb.sum()]])
                testes_rf.append({"resposta_fonte": rf, "modelo_a": a, "modelo_b": b,
                                  "taxa_a_%": round(100 * va.mean(), 2), "taxa_b_%": round(100 * vb.mean(), 2),
                                  "p_valor_fisher": float(f"{p:.3g}")})
        pd.DataFrame(testes_rf).to_csv(SAIDA / "testes_por_resposta_fonte.csv", index=False)
    except ImportError:
        pass

    # ---- Gráficos (cores validadas para daltonismo; hachuras para impressão em preto e branco)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    CORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    HACHURAS = ["", "//", "..", "xx"]
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    def barras(ax, pivo, inf=None, sup=None, legenda="upper right"):
        n = len(pivo.columns)
        larg = 0.8 / n
        x = np.arange(len(pivo.index))
        for i, modelo in enumerate(pivo.columns):
            pos = x - 0.4 + larg * (i + 0.5)
            erro = None
            if inf is not None:
                erro = [pivo[modelo] - inf[modelo], sup[modelo] - pivo[modelo]]
            ax.bar(pos, pivo[modelo], larg * 0.9, color=CORES[i % 4], hatch=HACHURAS[i % 4], edgecolor="white",
                   linewidth=0.5, label=modelo, yerr=erro, error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": "#444444"})
        ax.set_xticks(x)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.legend(title="Modelo", fontsize=8, title_fontsize=8, frameon=False, loc=legenda)

    pivo = tabela.pivot(index="mr", columns="modelo", values="taxa_violacao_%")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    barras(ax, pivo)
    ax.set_xlabel("Relação metamórfica")
    ax.set_ylabel("Taxa de violação (%)")
    ax.set_xticklabels([f"{mr}\n" + MRS[mr]["nome"].replace(" ", "\n", 1) for mr in pivo.index], fontsize=8)
    plt.tight_layout()
    plt.savefig(SAIDA / "grafico_violacoes.png", dpi=200)
    plt.close(fig)

    trf = pd.DataFrame(linhas_rf)
    piv = trf.pivot(index="resposta_fonte", columns="modelo", values="taxa_violacao_%").loc[["NAO", "SIM"]]
    inf = trf.pivot(index="resposta_fonte", columns="modelo", values="ic95_inf_%").loc[["NAO", "SIM"]]
    sup = trf.pivot(index="resposta_fonte", columns="modelo", values="ic95_sup_%").loc[["NAO", "SIM"]]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    barras(ax, piv, inf, sup, legenda="upper left")
    ax.set_xlabel("Resposta do modelo ao caso fonte")
    ax.set_ylabel("Taxa de violação (%)")
    ax.set_xticklabels(["NÃO", "SIM"])
    plt.tight_layout()
    plt.savefig(SAIDA / "grafico_por_resposta_fonte.png", dpi=200)
    plt.close(fig)

    print(pd.DataFrame(linhas_modelo).round(3).to_string(index=False))
    print()
    print(pd.DataFrame(linhas_vies).round(1).to_string(index=False))
    print()
    print(pd.DataFrame(linhas_rf).round(2).to_string(index=False))
    print()
    print(tabela.to_string(index=False))


if __name__ == "__main__":
    main()

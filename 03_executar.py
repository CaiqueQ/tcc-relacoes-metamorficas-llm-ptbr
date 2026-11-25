"""Passo 3: envia casos fonte e follow-up aos modelos e salva as respostas.

Uso:
    python 03_executar.py --piloto gpt-oss-120b    (só 10 entradas, salvas à parte em piloto.jsonl)
    python 03_executar.py gpt-oss-120b             (API gratuita do Groq)
    python 03_executar.py qwen3-4b                 (na GPU do Colab; idem qwen3-8b e tucano2-3.7b)

A execução é retomável: se cair no meio, ou se a cota diária gratuita acabar,
rode de novo (no dia seguinte, no caso da cota) que ela continua de onde parou.
Respostas ficam em resultados/respostas.jsonl (uma linha por chamada).

Análise de sensibilidade à instrução (opcional; não incluída nos resultados do TCC):
    python 03_executar.py --prompt p2 --so-fonte qwen3-4b   (só os 100 casos fonte: teste rápido)
    python 03_executar.py --prompt p2 qwen3-4b              (execução completa; reaproveita o que já rodou)
As respostas com a instrução p2 ficam em resultados/respostas_p2.jsonl; as originais não são tocadas.
"""
import json
import re
import sys
from datetime import datetime, timezone

import pandas as pd

import modelos
from config import DADOS, N_REPETICOES, PROMPTS, RESULTADOS, TODOS_OS_MODELOS, arquivo_respostas
from transformacoes import MRS, remover_acentos

N_PILOTO = 10


def interpretar(resposta: str) -> str:
    """Converte a resposta livre do modelo em SIM, NAO ou INVALIDA."""
    resposta = re.sub(r"<think>.*?</think>", "", resposta, flags=re.DOTALL)  # raciocínio dos Qwen3
    texto = remover_acentos(resposta).upper()
    m = re.search(r"\b(SIM|NAO)\b", texto)
    return m.group(1) if m else "INVALIDA"


def montar_entradas():
    fonte = pd.read_csv(DADOS / "casos_fonte.csv")
    followup = pd.read_csv(DADOS / "casos_followup.csv")

    # Casos de MRs revisadas manualmente só entram se aprovados
    revisao = pd.read_excel(DADOS / "revisao_manual.xlsx", dtype=str).fillna("")
    aprovados = revisao[revisao.aprovado.str.upper().str.strip() == "S"]
    correcoes = aprovados.set_index("id_caso")
    revisar = followup.mr.map(lambda c: MRS[c]["revisar"])
    followup = followup[~revisar | followup.id_caso.isin(aprovados.id_caso)].copy()
    for id_caso, linha in correcoes.iterrows():
        for campo in ("premissa", "hipotese"):
            if linha[f"correcao_{campo}"].strip():
                followup.loc[followup.id_caso == id_caso, campo] = linha[f"correcao_{campo}"].strip()

    print(f"Revisão manual: {len(aprovados)}/{len(revisao)} casos revisados aprovados")

    entradas = [{"id_caso": r.id_fonte, "premissa": r.premissa, "hipotese": r.hipotese} for r in fonte.itertuples()]
    entradas += [{"id_caso": r.id_caso, "premissa": r.premissa, "hipotese": r.hipotese} for r in followup.itertuples()]
    return entradas


def ja_feitas(arquivo):
    feitas = set()
    if arquivo.exists():
        with open(arquivo, encoding="utf-8") as f:
            for linha in f:
                r = json.loads(linha)
                feitas.add((r["modelo"], r["id_caso"], r["repeticao"]))
    return feitas


def main(nomes_modelos, piloto=False, prompt="p1", so_fonte=False):
    entradas = montar_entradas()
    arquivo = arquivo_respostas(prompt)
    if so_fonte:
        entradas = [e for e in entradas if "-" not in e["id_caso"]]  # casos fonte não têm sufixo -MRx
    if piloto:
        # 5 casos fonte + 5 subsequentes, para conferir respostas e tempo antes da execução completa
        entradas = entradas[:N_PILOTO // 2] + entradas[-(N_PILOTO // 2):]
        arquivo = RESULTADOS / "piloto.jsonl"
    feitas = ja_feitas(arquivo)
    for nome in nomes_modelos:
        gerar = modelos.carregar(TODOS_OS_MODELOS[nome])
        total = len(entradas) * N_REPETICOES
        print(f"\n== {nome}: {total} chamadas ==")
        with open(arquivo, "a", encoding="utf-8") as f:
            for i, e in enumerate(entradas):
                for rep in range(N_REPETICOES):
                    if (nome, e["id_caso"], rep) in feitas:
                        continue
                    texto = PROMPTS[prompt].format(premissa=e["premissa"], hipotese=e["hipotese"])
                    try:
                        resposta = gerar(texto)
                    except modelos.LimiteDiario:
                        feitas_agora = sum(1 for _ in open(arquivo, encoding="utf-8"))
                        sys.exit(f"\nA cota diária gratuita de {nome} acabou (entrada {i + 1}/{len(entradas)}). "
                                 f"Tudo o que foi feito está salvo ({feitas_agora} respostas no arquivo). "
                                 "Rode o mesmo comando amanhã: a execução continua de onde parou.")
                    f.write(json.dumps({
                        "modelo": nome, "id_caso": e["id_caso"], "repeticao": rep,
                        "resposta_bruta": resposta, "resposta": interpretar(resposta),
                        "horario": datetime.now(timezone.utc).isoformat(),
                    }, ensure_ascii=False) + "\n")
                    f.flush()
                print(f"  {i + 1}/{len(entradas)} entradas", end="\r")
        print()


def resumo_piloto():
    with open(RESULTADOS / "piloto.jsonl", encoding="utf-8") as f:
        for linha in f:
            r = json.loads(linha)
            print(f"{r['modelo']:<20} {r['id_caso']:<10} rep{r['repeticao']}  {r['resposta']:<9} bruta: {r['resposta_bruta'].strip()[:50]!r}")


if __name__ == "__main__":
    args = sys.argv[1:]
    piloto = "--piloto" in args
    so_fonte = "--so-fonte" in args
    prompt = "p1"
    if "--prompt" in args:
        i = args.index("--prompt")
        prompt = args[i + 1]
        del args[i:i + 2]
        if prompt not in PROMPTS:
            sys.exit(f"Instrução desconhecida: {prompt}. Opções: {list(PROMPTS)}")
    escolhidos = [a for a in args if a not in ("--piloto", "--so-fonte")]
    if not escolhidos:
        sys.exit(f"Informe o(s) modelo(s). Opções: {list(TODOS_OS_MODELOS)}")
    desconhecidos = set(escolhidos) - set(TODOS_OS_MODELOS)
    if desconhecidos:
        sys.exit(f"Modelos desconhecidos: {desconhecidos}. Opções: {list(TODOS_OS_MODELOS)}")
    main(escolhidos, piloto, prompt, so_fonte)
    if piloto:
        resumo_piloto()

"""Passo 0: testa a conexão com cada modelo enviando uma pergunta curta.

Uso:
    python 00_testar_conexao.py                   (testa o gerador e o gpt-oss-120b, pelo Groq)
    python 00_testar_conexao.py --listar          (lista os modelos disponíveis na sua conta do Groq)
    python 00_testar_conexao.py qwen3-4b          (testa um modelo local; precisa de GPU)
"""
import os
import sys
import time

import modelos
from config import GROQ_URL, MODELO_GERADOR, MODELOS, PROMPT_TAREFA, TODOS_OS_MODELOS


def listar_groq():
    from openai import OpenAI
    cliente = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url=GROQ_URL)
    ids = sorted(m.id for m in cliente.models.list().data)
    print("Modelos disponíveis na sua conta do Groq:")
    for i in ids:
        print("  ", i)
    return ids


if "--listar" in sys.argv:
    listar_groq()
    sys.exit()

PROMPT = PROMPT_TAREFA.format(premissa="Um menino está tocando violão.", hipotese="Um menino está tocando um instrumento.")

escolhidos = sys.argv[1:] or [n for n, c in MODELOS.items() if c["provedor"] != "local"]
alvos = [(n, TODOS_OS_MODELOS[n]) for n in escolhidos]
if not sys.argv[1:]:
    alvos.insert(0, ("gerador (" + MODELO_GERADOR["id"] + ")", MODELO_GERADOR))

falhas = 0
for nome, cfg in alvos:
    print(f"\n== {nome} ==")
    try:
        inicio = time.time()
        resposta = modelos.carregar(cfg)(PROMPT)
        print(f"OK em {time.time() - inicio:.1f}s. Resposta: {resposta.strip()[:80]!r}")
        print("   (esperado: SIM)")
    except KeyError as erro:
        falhas += 1
        print(f"FALHOU: chave {erro} não encontrada. Cadastre-a nos Secrets do Colab (ou no arquivo chaves.env).")
    except Exception as erro:
        falhas += 1
        print(f"FALHOU: {type(erro).__name__}: {str(erro)[:300]}")
        if "model_not_found" in str(erro) or "decommissioned" in str(erro):
            print("   O modelo não está disponível no Groq. Me envie a lista abaixo para escolhermos outro:")
            try:
                listar_groq()
            except Exception:
                pass
print("\nTudo certo!" if not falhas else f"\n{falhas} modelo(s) com problema. Confira a chave GROQ_API_KEY e o nome do modelo (python 00_testar_conexao.py --listar).")

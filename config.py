"""Configurações centrais do experimento.

Altere aqui os modelos, o tamanho da amostra e o número de repetições.
As chaves de API são lidas, nesta ordem:
    1. de variáveis de ambiente (GROQ_API_KEY; e, para os modelos pagos opcionais,
       OPENAI_API_KEY, ANTHROPIC_API_KEY, DASHSCOPE_API_KEY);
    2. do arquivo chaves.env nesta pasta (copie chaves.env.exemplo e preencha).
Nunca coloque chaves no código nem publique o arquivo chaves.env.
"""
import os
from pathlib import Path

RAIZ = Path(__file__).parent

_arquivo_chaves = RAIZ / "chaves.env"
if _arquivo_chaves.exists():
    for _linha in _arquivo_chaves.read_text(encoding="utf-8").splitlines():
        _linha = _linha.strip()
        if _linha and not _linha.startswith("#") and "=" in _linha:
            _nome, _valor = _linha.split("=", 1)
            _valor = _valor.strip().strip('"').strip("'")
            if _valor and not os.environ.get(_nome.strip()):
                os.environ[_nome.strip()] = _valor
DADOS = RAIZ / "dados"
RESULTADOS = RAIZ / "resultados"
DADOS.mkdir(exist_ok=True)
RESULTADOS.mkdir(exist_ok=True)

SEMENTE = 42
N_CASOS_FONTE = 100          # 50 ENTAILMENT + 50 NONE (amostra estratificada)
N_REPETICOES = 3             # execuções por entrada, para medir o não determinismo
TEMPERATURA = 0.0

# Relações metamórficas aplicadas. A MR8 (singular/plural) é opcional:
# inclua "MR8" na lista se o orientador aprovar.
MRS_ATIVAS = ["MR1", "MR2", "MR3", "MR4", "MR5", "MR6", "MR7"]

# ---------------------------------------------------------------------------
# Modelos sob teste (desenho 100% gratuito)
#   groq  -> API gratuita do Groq (limite diário; a execução retoma no dia seguinte)
#   local -> GPU gratuita do Colab
# ---------------------------------------------------------------------------
MODELOS = {
    # modelo aberto de grande porte da OpenAI; raciocínio interno no nível mais baixo
    "gpt-oss-120b": {"provedor": "groq", "id": "openai/gpt-oss-120b",
                     "extra": {"reasoning_effort": "low", "max_completion_tokens": 1024,
                               "extra_body": {"include_reasoning": False}}},
    # multilíngue de porte médio, quantizado em 4 bits para caber na GPU T4
    "qwen3-8b": {"provedor": "local", "id": "Qwen/Qwen3-8B", "quantizar_4bit": True,
                 "chat_kwargs": {"enable_thinking": False}},
    # multilíngue pequeno: a mesma família e porte de que o Tucano 2 deriva
    "qwen3-4b": {"provedor": "local", "id": "Qwen/Qwen3-4B",
                 "chat_kwargs": {"enable_thinking": False}},
    # especializado em português
    "tucano2-3.7b": {"provedor": "local", "id": "Polygl0t/Tucano2-qwen-3.7B-Instruct"},
}

# Modelos pagos, opcionais, para uma extensão futura do estudo.
# Para incluí-los: crie as chaves e rode, por exemplo, python 03_executar.py gpt-5.6-terra
MODELOS_PAGOS = {
    "gpt-5.6-terra": {"provedor": "openai", "id": "gpt-5.6-terra"},
    "claude-sonnet-5.5": {"provedor": "anthropic", "id": "claude-sonnet-5-5"},
    "qwen3-32b-api": {"provedor": "dashscope", "id": "qwen3-32b"},
}
TODOS_OS_MODELOS = {**MODELOS, **MODELOS_PAGOS}

# Modelo usado APENAS para gerar as transformações que exigem reescrita
# (MR1, MR2, MR4, MR7, MR8). Não está entre os modelos testados (é a versão menor do
# gpt-oss). Todas as saídas passam por revisão manual. Se o Groq desativar este modelo,
# rode "python 00_testar_conexao.py --listar" e troque o "id" por um dos disponíveis.
MODELO_GERADOR = {"provedor": "groq", "id": "openai/gpt-oss-20b", "intervalo": 4.0,  # ~15 chamadas/min
                  "extra": {"reasoning_effort": "low", "max_completion_tokens": 1024,
                            "extra_body": {"include_reasoning": False}}}

GROQ_URL = "https://api.groq.com/openai/v1"
DASHSCOPE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

# Instrução enviada aos modelos sob teste (igual para todos).
PROMPT_TAREFA = (
    "Leia as duas frases abaixo.\n\n"
    "Frase 1: {premissa}\n"
    "Frase 2: {hipotese}\n\n"
    "Considerando apenas o que a Frase 1 afirma, é possível concluir que a Frase 2 é verdadeira?\n"
    "Responda somente com SIM ou NÃO."
)

# Instruções alternativas para uma análise de sensibilidade opcional (não incluída nos resultados do TCC).
# "p1" é a instrução original; "p2" é menos restritiva e segue o critério de anotação do ASSIN 2,
# em que há implicação quando uma pessoa, ao ler a primeira frase, concluiria que a segunda é verdadeira.
PROMPTS = {
    "p1": PROMPT_TAREFA,
    "p2": (
        "Leia as duas frases abaixo.\n\n"
        "Frase 1: {premissa}\n"
        "Frase 2: {hipotese}\n\n"
        "Se a Frase 1 for verdadeira, uma pessoa que a lê concluiria que a Frase 2 também é verdadeira?\n"
        "Responda somente com SIM ou NÃO."
    ),
}


def arquivo_respostas(prompt="p1"):
    """p1 usa o arquivo original; as demais instruções gravam em arquivos separados."""
    return RESULTADOS / ("respostas.jsonl" if prompt == "p1" else f"respostas_{prompt}.jsonl")


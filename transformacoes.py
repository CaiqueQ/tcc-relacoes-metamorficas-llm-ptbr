"""Relações metamórficas (MRs) adaptadas ao português brasileiro.

Cada MR recebe um caso fonte (premissa, hipótese, rótulo) e devolve o caso
subsequente (follow-up) ou None quando a MR não se aplica àquele caso.

MRs de PRESERVAÇÃO (a resposta do modelo deve ser a mesma do caso fonte):
    MR1 Sinonímia               - reescrita por LLM + revisão manual
    MR2 Registro informal       - reescrita por LLM + revisão manual
    MR3 Escrita informal        - regra determinística (sem acentos, abreviações)
    MR4 Paráfrase sintática     - reescrita por LLM + revisão manual
    MR5 Variação regional       - regra determinística (léxico regional)
    MR7 Troca de gênero         - reescrita por LLM + revisão manual
    MR8 Singular/plural         - reescrita por LLM + revisão manual (opcional)
MR de INVERSÃO:
    MR6 Negação da hipótese     - regra determinística; se o modelo respondeu SIM
                                  no caso fonte, deve responder NÃO no follow-up

Correspondência com o catálogo MT4NLP (CHO; RUBERTO; TERRAGNI, 2025):
    MR1 ~ MRs 71-73 | MR3 ~ MRs 2, 29, 102, 125, 177 | MR4 ~ MR 136
    MR6 ~ MR 76     | MR7 ~ MR 75 | MR8 ~ MR 149 | MR2 e MR5: sem equivalente
"""
import json
import re
import unicodedata

PRESERVACAO = "preservacao"
INVERSAO = "inversao"

# "revisar": True -> o caso entra em dados/revisao_manual.csv e só é usado se aprovado
MRS = {
    "MR1": {"nome": "Sinonímia", "tipo": PRESERVACAO, "metodo": "llm", "revisar": True},
    "MR2": {"nome": "Registro informal", "tipo": PRESERVACAO, "metodo": "llm", "revisar": True},
    "MR3": {"nome": "Escrita informal", "tipo": PRESERVACAO, "metodo": "regra", "revisar": True},
    "MR4": {"nome": "Paráfrase sintática", "tipo": PRESERVACAO, "metodo": "llm", "revisar": True},
    "MR5": {"nome": "Variação regional", "tipo": PRESERVACAO, "metodo": "regra", "revisar": False},
    "MR6": {"nome": "Negação da hipótese", "tipo": INVERSAO, "metodo": "regra", "revisar": False},
    "MR7": {"nome": "Troca de gênero", "tipo": PRESERVACAO, "metodo": "llm", "revisar": True},
    "MR8": {"nome": "Singular/plural", "tipo": PRESERVACAO, "metodo": "llm", "revisar": True},
}


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------
def _substituir_palavras(texto: str, mapa: dict) -> str:
    """Substitui palavras inteiras preservando maiúscula inicial."""
    def troca(m):
        palavra = m.group(0)
        novo = mapa[palavra.lower()]
        return novo[0].upper() + novo[1:] if palavra[0].isupper() else novo

    padrao = r"\b(" + "|".join(re.escape(p) for p in sorted(mapa, key=len, reverse=True)) + r")\b"
    return re.sub(padrao, troca, texto, flags=re.IGNORECASE)


def remover_acentos(texto: str) -> str:
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


# --------------------------------------------------------------------------
# MR3 - Escrita informal (internetês)
# --------------------------------------------------------------------------
ABREVIACOES = {
    "você": "vc", "vocês": "vcs", "porque": "pq", "por que": "pq", "também": "tb",
    "muito": "mt", "muita": "mt", "para": "pra", "está": "tá", "estou": "tô",
    "estava": "tava", "não": "n", "que": "q", "quando": "qdo", "beleza": "blz",
}


def mr3_escrita_informal(premissa: str, hipotese: str):
    def informalizar(t):
        t = _substituir_palavras(t, ABREVIACOES)
        t = remover_acentos(t).lower()
        return t.rstrip(" .")

    return informalizar(premissa), informalizar(hipotese)


# --------------------------------------------------------------------------
# MR5 - Variação regional (léxico do português brasileiro)
# --------------------------------------------------------------------------
# Variantes lexicais regionais do português brasileiro, seguindo variações documentadas
# no Atlas Linguístico do Brasil (CARDOSO et al., 2014). Cada palavra é trocada por UMA
# variante de outra região. Só entram pares de MESMO gênero e número, para não quebrar a
# concordância com artigos e adjetivos (ex.: "a calçada" -> "o passeio" exigiria mudar o artigo).
REGIONALISMOS = {
    "menino": "guri", "meninos": "guris", "menina": "guria", "meninas": "gurias",
    "mandioca": "macaxeira", "mandiocas": "macaxeiras",
    "tangerina": "bergamota", "tangerinas": "bergamotas", "mexerica": "bergamota",
    "pipa": "pandorga", "pipas": "pandorgas",
    "semáforo": "farol", "semáforos": "faróis",
    "estilingue": "bodoque", "atiradeira": "baladeira",
    "pernilongo": "carapanã", "pernilongos": "carapanãs",
    "pão francês": "pão de sal",
}


def mr5_variacao_regional(premissa: str, hipotese: str):
    nova_p = _substituir_palavras(premissa, REGIONALISMOS)
    nova_h = _substituir_palavras(hipotese, REGIONALISMOS)
    if nova_p == premissa and nova_h == hipotese:
        return None  # nenhuma palavra com variante regional: MR não se aplica
    return nova_p, nova_h


# --------------------------------------------------------------------------
# MR6 - Negação da hipótese
# --------------------------------------------------------------------------
VERBOS_NEGAVEIS = [
    "está", "estão", "é", "são", "tem", "têm", "foi", "foram", "estava", "estavam",
    "vai", "vão", "pode", "podem", "há", "havia",
]
MARCAS_DE_NEGACAO = re.compile(r"\b(não|nenhum|nenhuma|ninguém|nada|nunca|jamais|sem)\b", re.IGNORECASE)


def mr6_negacao(premissa: str, hipotese: str):
    if MARCAS_DE_NEGACAO.search(hipotese):
        return None  # hipótese já negativa: dupla negação seria ambígua
    padrao = r"\b(" + "|".join(re.escape(v) for v in VERBOS_NEGAVEIS) + r")\b"
    m = re.search(padrao, hipotese, flags=re.IGNORECASE)
    if not m:
        return None
    nova_h = hipotese[: m.start()] + "não " + hipotese[m.start():]
    if m.start() == 0:  # verbo no início da frase
        nova_h = "Não " + hipotese[0].lower() + hipotese[1:]
    return premissa, nova_h


# --------------------------------------------------------------------------
# MR1, MR2, MR4 - reescritas feitas por LLM (revisadas manualmente depois)
# --------------------------------------------------------------------------
INSTRUCOES_LLM = {
    "MR1": (
        "Substitua uma ou duas palavras de conteúdo (substantivos, verbos ou adjetivos) "
        "por sinônimos usados no português brasileiro. Se a mesma palavra aparecer nas "
        "duas frases, use o mesmo sinônimo nas duas. Não altere mais nada."
    ),
    "MR2": (
        "Reescreva as frases no registro informal falado do português brasileiro "
        "(por exemplo: 'a gente' no lugar de 'nós', 'tá' no lugar de 'está', 'pra' no lugar de 'para', "
        "construções típicas da fala). Mantenha a ortografia correta e o mesmo significado."
    ),
    "MR4": (
        "Reescreva cada frase alterando apenas a estrutura sintática: troque a voz ativa "
        "pela passiva (ou o contrário) ou reordene os termos da oração. Mantenha as mesmas "
        "palavras de conteúdo e exatamente o mesmo significado."
    ),
    "MR7": (
        "Troque as palavras que designam pessoas pelo gênero oposto (menino -> menina, homem -> mulher, "
        "ele -> ela), fazendo exatamente a mesma troca nas duas frases e ajustando a concordância de "
        "artigos, adjetivos, pronomes e particípios. Se não houver pessoas com gênero marcado, "
        "devolva as frases sem nenhuma alteração."
    ),
    "MR8": (
        "Passe para o plural o sujeito das frases (o menino -> os meninos), fazendo exatamente a mesma "
        "troca nas duas frases e ajustando a concordância de artigos, adjetivos e verbos. Se o sujeito "
        "já estiver no plural, passe-o para o singular."
    ),
}

PROMPT_GERADOR = """Você vai transformar um par de frases em português brasileiro.

Regra da transformação: {instrucao}

Restrição obrigatória: o significado de cada frase deve continuar exatamente o mesmo.
Não acrescente nem remova informações.

Frase 1: {premissa}
Frase 2: {hipotese}

Responda apenas com um JSON no formato {{"frase1": "...", "frase2": "..."}}."""


def mr_por_llm(codigo_mr: str, premissa: str, hipotese: str, gerar):
    """`gerar` é uma função texto -> texto (ver modelos.py)."""
    prompt = PROMPT_GERADOR.format(instrucao=INSTRUCOES_LLM[codigo_mr], premissa=premissa, hipotese=hipotese)
    resposta = re.sub(r"<think>.*?</think>", "", gerar(prompt), flags=re.DOTALL)
    m = re.search(r"\{.*\}", resposta, flags=re.DOTALL)
    if not m:
        return None
    try:
        dados = json.loads(m.group(0))
        return dados["frase1"].strip(), dados["frase2"].strip()
    except (json.JSONDecodeError, KeyError):
        return None


def aplicar_mr(codigo_mr: str, premissa: str, hipotese: str, rotulo: str, gerar=None):
    """Aplica uma MR ao caso fonte. Retorna (premissa, hipotese) ou None."""
    if codigo_mr == "MR3":
        return mr3_escrita_informal(premissa, hipotese)
    if codigo_mr == "MR5":
        return mr5_variacao_regional(premissa, hipotese)
    if codigo_mr == "MR6":
        if rotulo != "ENTAILMENT":
            return None  # negar a hipótese só tem efeito previsível quando há implicação
        return mr6_negacao(premissa, hipotese)
    if codigo_mr in INSTRUCOES_LLM:
        if gerar is None:
            raise ValueError(f"{codigo_mr} precisa de um modelo gerador")
        resultado = mr_por_llm(codigo_mr, premissa, hipotese, gerar)
        if resultado == (premissa, hipotese):
            return None  # o gerador não transformou nada
        return resultado
    raise KeyError(codigo_mr)

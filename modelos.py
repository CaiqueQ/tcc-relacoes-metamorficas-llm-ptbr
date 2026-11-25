"""Clientes dos modelos. Todos expõem a mesma interface: gerar(prompt) -> str.

Provedores:
    groq      - API gratuita do Groq (compatível com a da OpenAI). Chave: GROQ_API_KEY
    local     - modelo aberto rodando na GPU (Colab), via Hugging Face Transformers
    openai, anthropic, dashscope - provedores pagos, opcionais (ver MODELOS_PAGOS em config.py)
"""
import os
import time

from config import DASHSCOPE_URL, GROQ_URL, TEMPERATURA


class LimiteDiario(Exception):
    """A cota diária gratuita acabou. Rode de novo amanhã: a execução continua de onde parou."""


def _com_retentativas(funcao, tentativas=5):
    def envolvida(prompt):
        for i in range(tentativas):
            try:
                return funcao(prompt)
            except LimiteDiario:
                raise
            except Exception as erro:  # limite por minuto, timeout, instabilidade
                if i == tentativas - 1:
                    raise
                espera = 2 ** (i + 1)
                print(f"  erro ({type(erro).__name__}: {str(erro)[:120]}); nova tentativa em {espera}s")
                time.sleep(espera)
    return envolvida


def _ritmo(intervalo):
    """Garante um intervalo mínimo entre chamadas (respeita o limite de chamadas por minuto)."""
    ultimo = [0.0]

    def esperar():
        falta = intervalo - (time.time() - ultimo[0])
        if falta > 0:
            time.sleep(falta)
        ultimo[0] = time.time()
    return esperar


def cliente_openai(id_modelo, base_url=None, chave_env="OPENAI_API_KEY", extra=None, intervalo=0.0):
    from openai import BadRequestError, OpenAI, RateLimitError

    cliente = OpenAI(api_key=os.environ[chave_env], base_url=base_url, max_retries=0)
    usa_temperatura = {"valor": True}
    esperar = _ritmo(intervalo)

    def chamar(args):
        """Faz a chamada; no limite POR MINUTO, espera e tenta de novo (não conta como falha)."""
        for _ in range(30):
            esperar()
            try:
                return cliente.chat.completions.create(**args)
            except RateLimitError as erro:
                texto = str(erro).lower()
                espera = float(erro.response.headers.get("retry-after", 0) or 0)
                if espera > 120 or "per day" in texto or "(rpd)" in texto or "(tpd)" in texto:
                    raise LimiteDiario(str(erro)[:200]) from erro
                espera = max(espera, 5)
                print(f"  limite por minuto do Groq; aguardando {espera:.0f}s (normal)")
                time.sleep(espera)
        raise RuntimeError("limite por minuto persistente; rode a célula de novo em alguns minutos")

    def gerar(prompt):
        args = dict(model=id_modelo, messages=[{"role": "user", "content": prompt}], **(extra or {}))
        if usa_temperatura["valor"]:
            args["temperature"] = TEMPERATURA
        try:
            r = chamar(args)
        except BadRequestError as erro:
            # alguns modelos de raciocínio não aceitam temperatura; registra e segue sem ela
            if "temperature" in str(erro) and usa_temperatura["valor"]:
                print(f"  {id_modelo} não aceita 'temperature'; usando o padrão do modelo")
                usa_temperatura["valor"] = False
                args.pop("temperature")
                r = chamar(args)
            else:
                raise
        return r.choices[0].message.content or ""

    return _com_retentativas(gerar)


def cliente_anthropic(id_modelo):
    import anthropic

    cliente = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def gerar(prompt):
        r = cliente.messages.create(
            model=id_modelo, max_tokens=300, temperature=TEMPERATURA,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in r.content if b.type == "text")

    return _com_retentativas(gerar)


def cliente_local(id_modelo, quantizar_4bit=False, chat_kwargs=None, max_novos=20):
    """Modelos abertos via Hugging Face Transformers (Colab com GPU T4)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Nenhuma GPU encontrada. No Colab: Ambiente de execução > Alterar o tipo > GPU T4.")
    opcoes = dict(device_map="auto")
    if quantizar_4bit:
        from transformers import BitsAndBytesConfig
        opcoes["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    else:
        opcoes["torch_dtype"] = torch.float16
    print(f"  carregando {id_modelo} (na primeira vez, o download leva alguns minutos)...")
    tokenizador = AutoTokenizer.from_pretrained(id_modelo)
    modelo = AutoModelForCausalLM.from_pretrained(id_modelo, **opcoes)

    def gerar(prompt):
        texto = tokenizador.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, **(chat_kwargs or {}))
        entrada = tokenizador(texto, return_tensors="pt", add_special_tokens=False).to(modelo.device)
        with torch.no_grad():
            saida = modelo.generate(**entrada, max_new_tokens=max_novos, do_sample=False,
                                    pad_token_id=tokenizador.eos_token_id)  # decodificação gulosa
        return tokenizador.decode(saida[0][entrada["input_ids"].shape[1]:], skip_special_tokens=True)

    return gerar


def carregar(cfg):
    provedor, id_modelo = cfg["provedor"], cfg["id"]
    if provedor == "groq":
        return cliente_openai(id_modelo, base_url=GROQ_URL, chave_env="GROQ_API_KEY",
                              extra=cfg.get("extra"), intervalo=cfg.get("intervalo", 2.2))
    if provedor == "local":
        return cliente_local(id_modelo, cfg.get("quantizar_4bit", False), cfg.get("chat_kwargs"))
    if provedor == "openai":
        return cliente_openai(id_modelo, extra=cfg.get("extra"))
    if provedor == "anthropic":
        return cliente_anthropic(id_modelo)
    if provedor == "dashscope":
        return cliente_openai(id_modelo, base_url=DASHSCOPE_URL, chave_env="DASHSCOPE_API_KEY",
                              extra={"extra_body": {"enable_thinking": False}})
    raise ValueError(provedor)

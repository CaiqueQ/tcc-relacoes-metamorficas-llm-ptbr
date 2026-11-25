# Relações metamórficas para testes de robustez de LLMs em português brasileiro

Código, casos de teste e respostas dos modelos do Trabalho de Conclusão de Curso
**"Adaptação e Avaliação de Relações Metamórficas para Testes de Robustez de Grandes Modelos de
Linguagem em Português Brasileiro"**, apresentado ao Bacharelado em Ciências Exatas e Tecnológicas do
Centro de Ciências Exatas e Tecnológicas (CETEC) da Universidade Federal do Recôncavo da Bahia (UFRB).

- **Autor:** Caíque Queiroz Bandeira de Melo
- **Orientadora:** Profa. Dra. Camila Bezerra da Silva
- **Ano:** 2026

## Sobre o experimento

Sete relações metamórficas (MRs) foram aplicadas a 100 pares de frases da partição de teste do
corpus ASSIN 2 (inferência textual), sorteados de forma estratificada (50 com implicação e 50 sem).
Cada entrada foi enviada três vezes a quatro modelos de pesos abertos, sem custo. A resposta
considerada foi a do voto majoritário. Em seguida, compararam-se as respostas (SIM/NÃO) do caso fonte
e do caso transformado.

| MR | Transformação | Como é gerada | Relação esperada |
|---|---|---|---|
| MR1 | Sinonímia | LLM + revisão manual | mesma resposta |
| MR2 | Registro informal falado | LLM + revisão manual | mesma resposta |
| MR3 | Escrita informal (sem acentos, abreviações) | regra + revisão manual | mesma resposta |
| MR4 | Paráfrase sintática (voz ativa/passiva) | LLM + revisão manual | mesma resposta |
| MR5 | Variação regional do léxico | regra | mesma resposta |
| MR6 | Negação da hipótese | regra | se a fonte foi SIM, o caso subsequente deve ser NÃO |
| MR7 | Troca de gênero com concordância | LLM + revisão manual | mesma resposta |

As reescritas por LLM foram feitas pelo gpt-oss-20b, que não está entre os modelos testados. Todas
passaram por revisão manual: de 525 casos gerados, 490 foram usados (`resultados/tabela_revisao.csv`).

| Modelo | Identificador | Onde rodou |
|---|---|---|
| gpt-oss-120b | `openai/gpt-oss-120b` | API gratuita do Groq |
| Qwen3-8B (4 bits) | `Qwen/Qwen3-8B` | Google Colab, GPU T4 |
| Qwen3-4B | `Qwen/Qwen3-4B` | Google Colab, GPU T4 |
| Tucano 2 | `Polygl0t/Tucano2-qwen-3.7B-Instruct` | Google Colab, GPU T4 |

Os experimentos foram executados de 29 a 30 de outubro de 2026.

### Resultados principais

Nas relações de preservação (MR1 a MR5 e MR7), observaram-se 69 violações em 1.764 pares avaliados.
A MR6 não teve violações.

| Modelo | Acurácia nos casos fonte | Violações nas relações de preservação |
|---|---|---|
| gpt-oss-120b | 88% | 20 de 441 (4,5%) |
| Qwen3-4B | 61% | 23 de 441 (5,2%) |
| Qwen3-8B | 61% | 17 de 441 (3,9%) |
| Tucano 2 | 56% | 9 de 441 (2,0%) |

A discussão completa, com intervalos de confiança, testes estatísticos e a análise por resposta ao
caso fonte, está no Capítulo 4 do TCC.

## Estrutura do repositório

```
├── 00_testar_conexao.py       testa a chave do Groq e os modelos acessados por API
├── 01_preparar_dados.py       baixa o ASSIN 2 e sorteia os 100 casos fonte
├── 02_gerar_casos.py          aplica as MRs e gera a planilha de revisão manual
├── 03_executar.py             envia os casos aprovados aos modelos e registra as respostas
├── 04_analisar.py             calcula as métricas e gera as tabelas e os gráficos
├── 05_comparar_instrucoes.py  análise de sensibilidade à instrução (opcional; fora do TCC)
├── config.py                  modelos, MRs ativas, amostra, repetições e instrução
├── modelos.py                 clientes da API e execução local dos modelos
├── transformacoes.py          implementação das relações metamórficas
├── experimento_colab.ipynb    notebook para rodar tudo no Google Colab
├── requirements.txt
├── chaves.env.exemplo         modelo do arquivo de chaves (o arquivo real não é publicado)
├── dados/
│   ├── casos_fonte.csv        100 pares sorteados do ASSIN 2
│   ├── casos_followup.csv     casos subsequentes gerados por MR
│   ├── revisao_manual.xlsx    revisão manual (coluna "aprovado", correções e motivos)
│   └── cache_gerador.jsonl    saídas brutas do modelo gerador
└── resultados/
    ├── respostas.jsonl        respostas brutas dos modelos, com data e hora
    ├── piloto.jsonl           respostas da execução piloto
    └── tabelas (.csv) e gráficos (.png) do Capítulo 4
```

### Correspondência com o TCC

| Arquivo | Elemento do TCC |
|---|---|
| `resultados/tabela_revisao.csv` | Tabela 1 |
| `resultados/tabela_modelos.csv` e `tabela_vies.csv` | Tabela 2 |
| `resultados/tabela_violacoes.csv` | Tabela 3 |
| `resultados/grafico_violacoes.png` | Figura 2 |
| `resultados/testes_estatisticos.csv` | Tabela 4 |
| `resultados/tabela_violacoes_por_resposta_fonte.csv` | Tabela 5 |
| `resultados/grafico_por_resposta_fonte.png` | Figura 3 |
| `resultados/testes_por_resposta_fonte.csv` | Tabela 6 |
| `resultados/violacoes_detalhadas.csv` | Quadro 6 e Seção 4.5 |

## Como reproduzir

### Só a análise (sem chamar nenhum modelo)

Os dados e as respostas já estão no repositório, então as tabelas e os gráficos podem ser recalculados
em poucos segundos:

```bash
pip install -r requirements.txt
python 04_analisar.py
```

### O experimento completo

O caminho recomendado é o Google Colab, com o notebook `experimento_colab.ipynb`. Copie a pasta do
repositório para o seu Google Drive com o nome `tcc`, cadastre a chave `GROQ_API_KEY` nos Secrets do
Colab e rode as células de cima para baixo.

| Passo | Comando | O que faz |
|---|---|---|
| 0 | `python 00_testar_conexao.py` | Testa a chave do Groq |
| 1 | `python 01_preparar_dados.py` | Baixa o ASSIN 2 e sorteia os 100 casos fonte (semente 42) |
| 2 | `python 02_gerar_casos.py` | Aplica as MRs e cria `dados/revisao_manual.xlsx` |
| — | revisão manual | Preencha a coluna `aprovado` (S/N) da planilha |
| 3 | `python 03_executar.py <modelo>` | Executa um modelo (retoma de onde parou) |
| 4 | `python 04_analisar.py` | Gera as tabelas e os gráficos em `resultados/` |

Os modelos locais precisam de GPU (no Colab, ambiente GPU T4) e das bibliotecas
`torch transformers accelerate bitsandbytes`. O gpt-oss-120b usa a cota gratuita diária do Groq, e a
execução completa pode levar de 2 a 3 dias.

Observações:

- Os passos 1 e 2 sobrescrevem os arquivos de `dados/`. Para refazer só a execução ou a análise com os
  casos usados no TCC, comece do passo 3 ou 4.
- Mesmo com temperatura 0 e decodificação gulosa, os modelos podem responder de forma diferente em
  novas execuções, e os modelos acessados por API podem ser atualizados ou desativados pelo provedor.

### Chaves de API

A única chave obrigatória é a `GROQ_API_KEY`, que é gratuita e pode ser criada em https://console.groq.com/keys.
No seu computador, copie `chaves.env.exemplo` para `chaves.env` e preencha. O arquivo `chaves.env`
está no `.gitignore` e **nunca deve ser publicado**. As chaves dos modelos pagos (`MODELOS_PAGOS` em
`config.py`) são opcionais e não foram usadas no TCC.

## Dados

Os casos fonte vêm da partição de teste do ASSIN 2 (REAL et al., 2020), disponível em
https://huggingface.co/datasets/nilc-nlp/assin2 e https://github.com/ruanchaves/assin.
Ao usar estes dados, cite também o corpus original:

> REAL, L.; FONSECA, E.; GONÇALO OLIVEIRA, H. The ASSIN 2 shared task: a quick overview. In:
> INTERNATIONAL CONFERENCE ON COMPUTATIONAL PROCESSING OF THE PORTUGUESE LANGUAGE, 14., 2020, Évora.
> Proceedings [...]. Cham: Springer, 2020. p. 406-412. (Lecture Notes in Computer Science, v. 12037).

## Como citar

> MELO, Caíque Queiroz Bandeira de. **Adaptação e avaliação de relações metamórficas para testes de
> robustez de grandes modelos de linguagem em português brasileiro**. 2026. Trabalho de Conclusão de
> Curso (Bacharelado em Ciências Exatas e Tecnológicas) – Centro de Ciências Exatas e Tecnológicas,
> Universidade Federal do Recôncavo da Bahia, Cruz das Almas, 2026.

## Licença

O código está sob a licença MIT (arquivo `LICENSE`). Os dados derivados do ASSIN 2 seguem os termos do
corpus original.

"""Passo 1: baixa o ASSIN 2 (teste) e sorteia a amostra estratificada de casos fonte."""
import urllib.request
import xml.etree.ElementTree as ET

import pandas as pd

from config import DADOS, N_CASOS_FONTE, SEMENTE

URL_XML = "https://raw.githubusercontent.com/ruanchaves/assin/master/sources/assin2-test.xml"


def pelo_hugging_face():
    from datasets import load_dataset
    ds = load_dataset("nilc-nlp/assin2", split="test")
    df = ds.to_pandas()
    nomes = ds.features["entailment_judgment"].names  # ["NONE", "ENTAILMENT"]
    df["rotulo"] = df["entailment_judgment"].map(lambda i: nomes[i])
    return df.rename(columns={"premise": "premissa", "hypothesis": "hipotese", "sentence_pair_id": "id_par"})


def pelo_github():
    """Plano B: o XML original do ASSIN 2, mantido no GitHub."""
    with urllib.request.urlopen(URL_XML, timeout=60) as r:
        raiz = ET.fromstring(r.read())
    linhas = [{"id_par": int(par.get("id")), "premissa": par.find("t").text.strip(),
               "hipotese": par.find("h").text.strip(), "rotulo": par.get("entailment").upper()}
              for par in raiz.iter("pair")]
    return pd.DataFrame(linhas)


try:
    df = pelo_hugging_face()
    print("ASSIN 2 carregado pelo Hugging Face")
except Exception as erro:
    print(f"Hugging Face indisponível ({type(erro).__name__}); baixando o XML pelo GitHub...")
    df = pelo_github()
print(f"Partição de teste: {len(df)} pares")

por_classe = N_CASOS_FONTE // 2
amostra = (
    df.groupby("rotulo")
    .sample(n=por_classe, random_state=SEMENTE)
    .sort_values(["rotulo", "id_par"], ascending=[False, True])
    .reset_index(drop=True)
)
amostra.insert(0, "id_fonte", [f"F{i:03d}" for i in range(len(amostra))])
amostra[["id_fonte", "id_par", "premissa", "hipotese", "rotulo"]].to_csv(DADOS / "casos_fonte.csv", index=False)

print(f"{len(amostra)} casos fonte salvos em {DADOS / 'casos_fonte.csv'}")
print(amostra["rotulo"].value_counts())

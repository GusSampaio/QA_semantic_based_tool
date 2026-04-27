import spacy
from src.auxiliares import normalizar_termo

def extrair_copula(frase: str, nlp: spacy.Language) -> list[tuple]:
    doc = nlp(frase)
    triplas = []

    for token in doc:
        # Passo 1: encontra o "é", "são", "era"... (dep == cop)
        if token.dep_ != "cop":
            continue

        # Passo 2: o head desse token é o predicativo
        predicativo = token.head

        # Passo 3: busca o sujeito — irmão do predicativo com dep == nsubj
        sujeito_token = None
        for irmao in predicativo.head.children if predicativo.dep_ == "ROOT" else predicativo.children:
            if irmao.dep_ == "nsubj":
                sujeito_token = irmao
                break

        # Fallback: busca nsubj diretamente nos filhos do predicativo
        if sujeito_token is None:
            for filho in predicativo.children:
                if filho.dep_ == "nsubj":
                    sujeito_token = filho
                    break

        if sujeito_token is None:
            continue

        # Monta os spans normalizados
        sujeito = normalizar_termo(
            " ".join(t.text for t in sujeito_token.subtree)
        )
        objeto = normalizar_termo(
            " ".join(t.text for t in predicativo.subtree
                     # exclui o sujeito e a cópula do span do predicativo
                     if t not in list(sujeito_token.subtree)
                     and t.dep_ != "cop"
                     and t.dep_ != "nsubj")
        )

        if not sujeito or not objeto:
            continue

        # Passo 4: diferencia o predicado pelo POS do predicativo
        if predicativo.pos_ in ("NOUN", "PROPN"):
            predicado = "instancia_de"
        elif predicativo.pos_ == "ADJ":
            predicado = "tem_propriedade"
        else:
            predicado = "ser"

        triplas.append((sujeito, predicado, objeto, "regra_copula"))

    return triplas
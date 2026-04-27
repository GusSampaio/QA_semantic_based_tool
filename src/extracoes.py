import spacy
from src.regras import extrair_copula

def extrair_triplas(frases: list, nlp: spacy.Language) -> list[tuple]:
    triplas = []

    for frase in frases:
        triplas.extend(extrair_copula(frase.strip(), nlp))

    # Remove duplicatas por (sujeito, predicado, objeto)
    vistas = set()
    triplas_unicas = []
    for sujeito, predicado, objeto, origem in triplas:
        chave = (sujeito, predicado, objeto)
        if chave not in vistas:
            vistas.add(chave)
            triplas_unicas.append((sujeito, predicado, objeto, origem))

    return triplas_unicas
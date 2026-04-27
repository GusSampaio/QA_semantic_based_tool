import spacy
import src.regras as regras_module
# from src.regras import extrair_copula

def extrair_triplas(frases: list, nlp: spacy.Language) -> list[tuple]:
    triplas = []

    for frase in frases:
        doc = nlp(frase.strip())
        triplas.extend(regras_module.extrair_copula(frase.strip(), doc))
        triplas.extend(regras_module.extrair_objeto_direto(frase.strip(), doc))

    # Remove duplicatas por (sujeito, predicado, objeto)
    vistas = set()
    triplas_unicas = []
    for sujeito, predicado, objeto, origem in triplas:
        chave = (sujeito, predicado, objeto)
        if chave not in vistas:
            vistas.add(chave)
            triplas_unicas.append((sujeito, predicado, objeto, origem))

    return triplas_unicas
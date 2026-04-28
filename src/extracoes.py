import spacy
import src.regras as regras_module
import src.frames as frames_module

def extrair_triplas_regras(frases: list, nlp: spacy.Language) -> list[tuple]:
    triplas = []

    for frase in frases:
        doc = nlp(frase.strip())
        triplas.extend(regras_module.extrair_copula(doc))
        triplas.extend(regras_module.extrair_objeto_direto(doc))
        triplas.extend(regras_module.extrair_obl(doc))

    # Remove duplicatas por (sujeito, predicado, objeto)
    vistas = set()
    triplas_unicas = []
    for sujeito, predicado, objeto, origem in triplas:
        chave = (sujeito, predicado, objeto)
        if chave not in vistas:
            vistas.add(chave)
            triplas_unicas.append((sujeito, predicado, objeto, origem))

    return triplas_unicas

def extrair_triplas_frames(frases: list, nlp: spacy.Language) -> list[tuple]:
    triplas = []

    for frase in frases:
        doc = nlp(frase.strip())

        frames = frames_module.extrair_todos_frames(doc)
        triplas.extend(frames_module.frames_para_triplas(frames))

    # Remove duplicatas
    vistas = set()
    triplas_unicas = []
    for sujeito, predicado, objeto in triplas:
        chave = (sujeito, predicado, objeto)
        if chave not in vistas:
            vistas.add(chave)
            triplas_unicas.append((sujeito, predicado, objeto, "frame_evento"))

    return triplas_unicas
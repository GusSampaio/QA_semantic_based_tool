import spacy
import src.frames as frames_module

def extrair_triplas_frames(frases: list, nlp: spacy.Language) -> list:
    elementos = []
    event_id = 0

    for frase in frases:
        doc = nlp(frase.strip())
        frames = frames_module.extrair_todos_frames(doc)
        novos_elementos, event_id = frames_module.frames_para_grafo_estruturado(frames, event_id)
        elementos.extend(novos_elementos)

    return elementos


def extrair_triplas_frames_com_metodo(
    frases: list, nlp: spacy.Language, metodo: str = "simbolico"
) -> list:
    """Extrai frames usando o método escolhido pelo usuário.

    "simbolico" → regras linguísticas sobre Universal Dependencies (padrão).
    "srl"       → modelo neural BERT de Semantic Role Labeling.
    """
    from src.extrator_base import criar_extrator

    return criar_extrator(metodo).extrair_elementos(frases, nlp)
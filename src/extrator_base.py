"""Interface comum e fábrica dos extratores de frames semânticos.

Todo extrator — simbólico ou neural — implementa `extrair_elementos` e devolve
elementos no formato consumido por `grafo.construir_grafo`:
    nó    : {"tipo": "no",     "id": str, "attrs": {"tipo_evento": str}}
    aresta: {"tipo": "aresta", "origem": str, "destino": str, "papel": str}
"""

from abc import ABC, abstractmethod


class ExtratorBase(ABC):
    @abstractmethod
    def extrair_elementos(self, frases: list, nlp) -> list:
        """Extrai nós e arestas do grafo a partir de sentenças já segmentadas.

        Parâmetros
        ----------
        frases : list[str]
            Sentenças limpas (saída de `auxiliares.separar_frases`).
        nlp : spacy.Language
            Modelo spaCy já carregado.
        """
        ...


class ExtratorSimbolico(ExtratorBase):
    """Pipeline simbólica original (regras sobre labels UD em `frames.py`)."""

    def extrair_elementos(self, frases: list, nlp) -> list:
        import src.frames as frames_module

        elementos = []
        event_id = 0
        for frase in frases:
            doc = nlp(frase.strip())
            frames = frames_module.extrair_todos_frames(doc)
            novos, event_id = frames_module.frames_para_grafo_estruturado(
                frames, event_id
            )
            elementos.extend(novos)
        return elementos


def criar_extrator(metodo: str) -> ExtratorBase:
    """Fábrica de extratores.

    "simbolico" → regras linguísticas sobre Universal Dependencies (padrão).
    "srl"       → modelo neural BERT de Semantic Role Labeling.
    """
    if metodo == "srl":
        from src.extrator_srl import ExtratorSRL

        return ExtratorSRL()
    return ExtratorSimbolico()

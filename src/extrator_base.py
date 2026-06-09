"""
Interface base para extratores de frames semânticos.

Para adicionar uma nova abordagem de extração:
  1. Crie uma subclasse de ExtratorBase implementando extrair_elementos().
  2. Registre o nome do método em criar_extrator().
"""

from abc import ABC, abstractmethod


class ExtratorBase(ABC):
    """Interface comum a todos os extratores de frames semânticos.

    Qualquer extrator — simbólico, estatístico ou híbrido — deve
    implementar extrair_elementos() e retornar elementos no formato
    compatível com grafo.construir_grafo().
    """

    @abstractmethod
    def extrair_elementos(self, frases: list, nlp) -> list:
        """Extrai nós e arestas do grafo a partir de sentenças.

        Parâmetros
        ----------
        frases : list[str]
            Sentenças limpas e segmentadas (saída de auxiliares.separar_frases).
        nlp : spacy.Language
            Modelo spaCy já carregado.

        Retorna
        -------
        list[dict]
            Lista de elementos no formato esperado por grafo.construir_grafo():
            - Nó:    {"tipo": "no",    "id": str, "attrs": {"tipo_evento": str}}
            - Aresta: {"tipo": "aresta", "origem": str, "destino": str, "papel": str}
        """
        pass


class ExtratorSimbolico(ExtratorBase):
    """Adapta a pipeline simbólica original (frames.py) à interface ExtratorBase.

    Não altera nenhuma lógica de frames.py; apenas repassa as chamadas
    para manter retrocompatibilidade total.
    """

    def extrair_elementos(self, frases: list, nlp) -> list:
        import src.frames as frames_module  # import local evita dependência circular

        elementos = []
        event_id = 0
        for frase in frases:
            doc = nlp(frase.strip())
            frames = frames_module.extrair_todos_frames(doc)
            novos_elementos, event_id = frames_module.frames_para_grafo_estruturado(
                frames, event_id
            )
            elementos.extend(novos_elementos)
        return elementos


def criar_extrator(metodo: str) -> ExtratorBase:
    """Fábrica de extratores.

    Parâmetros
    ----------
    metodo : str
        "simbolico"   → regras linguísticas sobre labels UD (padrão).
        "estatistico" → pontuação por frequência e distância na árvore.

    Retorna
    -------
    ExtratorBase
        Instância pronta para uso.
    """
    if metodo == "estatistico":
        from src.extrator_estatistico import ExtratorEstatistico
        return ExtratorEstatistico()
    return ExtratorSimbolico()

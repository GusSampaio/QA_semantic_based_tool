"""Extração de frames semânticos via modelo neural de Semantic Role Labeling.

Usa o modelo `GusSampaio/bert-base-portuguese-cased-srl` (BertForTokenClassification,
rótulos estilo PropBank). O modelo exige que o predicado venha marcado com os tokens
especiais `<PRED>` … `</PRED>` e processa um predicado por vez — por isso o spaCy
ainda é usado para localizar os predicados (verbos e cópulas) na frase.

Os rótulos do modelo mapeiam diretamente na estrutura de frame do projeto:
    ARG0 → Arg0, ARG1 → Arg1, ARG2 → Arg2,
    ARGM-LOC → loc, ARGM-TMP/ARGM-TML → tmp, demais ARGM-* → outros.
"""

from functools import lru_cache

from src.auxiliares import normalizar_termo
from src.extrator_base import ExtratorBase
import src.frames as frames_module


MODELO_SRL = "GusSampaio/bert-base-portuguese-cased-srl"

_PAPEIS_IGNORADOS = frozenset({"O", "PRED", "INC"})
_ARGM_LOC = frozenset({"ARGM-LOC"})
_ARGM_TMP = frozenset({"ARGM-TMP", "ARGM-TML"})


class ExtratorSRL(ExtratorBase):
    """Extrator neural de frames. Saída idêntica à do extrator simbólico."""

    def extrair_elementos(self, frases: list, nlp) -> list:
        pipe = _carregar_pipeline()

        elementos = []
        event_id = 0
        for frase in frases:
            frase = frase.strip()
            if not frase:
                continue

            doc = nlp(frase)
            frames = []
            for token in doc:
                if not _eh_predicado(token):
                    continue
                entrada = _marcar_predicado(frase, token)
                papeis = _agrupar_papeis(pipe(entrada), entrada)
                frame = _frame_do_predicado(token, papeis)
                if frame["Arg0"] or frame["Arg1"]:
                    frames.append(frame)

            novos, event_id = frames_module.frames_para_grafo_estruturado(
                frames, event_id
            )
            elementos.extend(novos)

        return elementos


def _eh_predicado(token) -> bool:
    """Verbos e cópulas — espelha os frames verbais e copulativos do simbólico."""
    return token.pos_ == "VERB" or token.dep_ == "cop"


def _marcar_predicado(frase: str, token) -> str:
    """Insere os tokens especiais <PRED> … </PRED> ao redor do predicado."""
    inicio = token.idx
    fim = token.idx + len(token.text)
    return f"{frase[:inicio]}<PRED> {frase[inicio:fim]} </PRED>{frase[fim:]}"


def _agrupar_papeis(saida_pipeline: list, entrada: str) -> dict:
    """Agrupa a saída do pipeline por papel semântico → lista de spans normalizados.

    O span é recortado da string de entrada via offsets `start`/`end` (e não do
    campo `word`, sujeito a artefatos de wordpiece) e expandido até a fronteira da
    palavra, pois o modelo rotula no nível de subpalavra e às vezes corta no meio
    do token ortográfico (ex.: "mitose" → "mitos", "células-filhas" → "células").
    """
    papeis: dict = {}
    for grupo in saida_pipeline:
        papel = grupo["entity_group"]
        if papel in _PAPEIS_IGNORADOS:
            continue
        ini, fim = _expandir_para_palavra(entrada, grupo["start"], grupo["end"])
        texto = normalizar_termo(
            entrada[ini:fim].replace("<PRED>", "").replace("</PRED>", "")
        )
        if texto:
            papeis.setdefault(papel, []).append(texto)
    return papeis


def _expandir_para_palavra(texto: str, ini: int, fim: int) -> tuple:
    """Estende [ini, fim) para abranger palavras inteiras (letras, dígitos, hífen)."""
    def faz_parte(c: str) -> bool:
        return c.isalnum() or c == "-"

    while ini > 0 and faz_parte(texto[ini - 1]):
        ini -= 1
    while fim < len(texto) and faz_parte(texto[fim]):
        fim += 1
    return ini, fim


def _frame_do_predicado(token, papeis: dict) -> dict:
    """Converte os papéis rotulados de um predicado na estrutura de frame do projeto."""
    argms = {"loc": [], "tmp": [], "outros": []}
    for papel, spans in papeis.items():
        if not papel.startswith("ARGM-"):
            continue
        if papel in _ARGM_LOC:
            argms["loc"].extend(spans)
        elif papel in _ARGM_TMP:
            argms["tmp"].extend(spans)
        else:
            argms["outros"].extend(spans)

    def primeiro(papel: str):
        spans = papeis.get(papel)
        return spans[0] if spans else None

    return {
        "verbo": token,
        "predicado": normalizar_termo(token.lemma_),
        "Arg0": primeiro("ARG0"),
        "Arg1": primeiro("ARG1"),
        "Arg2": primeiro("ARG2"),
        "ArgMs": argms,
    }


@lru_cache(maxsize=1)
def _carregar_pipeline():
    """Carrega o pipeline de SRL uma única vez por processo (lazy)."""
    from transformers import pipeline

    return pipeline(
        "token-classification",
        model=MODELO_SRL,
        aggregation_strategy="simple",
    )

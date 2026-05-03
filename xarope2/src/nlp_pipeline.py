"""Integration layer for the symbolic NLP/frame/graph pipeline.

This module keeps the first student's spaCy + frame extraction logic reusable
outside the original Streamlit-only app. It converts extracted frames into two
forms:

1. graph elements, used for NetworkX visualization; and
2. symbolic facts, stored as vector-searchable LangChain documents for RAG.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import networkx as nx
import pandas as pd
import spacy
if TYPE_CHECKING:
    from langchain_core.documents import Document

from src.auxiliares import limpar_texto, normalizar_termo, separar_frases
from src import frames as frames_module
from src import grafo as grafo_module


@dataclass(frozen=True)
class SymbolicFact:
    """Structured fact derived from a semantic frame."""

    event_id: str
    predicate: str
    subject: str | None
    object: str | None
    arg2: str | None
    locations: tuple[str, ...]
    times: tuple[str, ...]
    modifiers: tuple[str, ...]
    sentence: str

    @property
    def relation_label(self) -> str:
        if self.predicate == "instancia_de":
            return "é / instancia_de"
        if self.predicate == "tem_propriedade":
            return "tem propriedade"
        return self.predicate

    def as_text(self) -> str:
        """Render a fact as context suitable for embedding and LLM prompts."""
        roles: list[str] = []
        if self.subject:
            roles.append(f"Arg0/agente/sujeito={self.subject}")
        if self.object:
            roles.append(f"Arg1/paciente/objeto={self.object}")
        if self.arg2:
            roles.append(f"Arg2={self.arg2}")
        if self.locations:
            roles.append("local=" + ", ".join(self.locations))
        if self.times:
            roles.append("tempo=" + ", ".join(self.times))
        if self.modifiers:
            roles.append("modificadores=" + ", ".join(self.modifiers))

        if self.subject and self.object:
            triple = f"{self.subject} --[{self.predicate}]--> {self.object}"
        elif self.object:
            triple = f"evento {self.event_id} --[{self.predicate}]--> {self.object}"
        elif self.subject:
            triple = f"{self.subject} --[{self.predicate}]--> evento {self.event_id}"
        else:
            triple = f"evento {self.event_id} --[{self.predicate}]"

        rendered_roles = "; ".join(roles) if roles else "sem papéis explícitos"
        return (
            f"Fato simbólico extraído por NLP: {triple}\n"
            f"Papéis semânticos: {rendered_roles}\n"
            f"Frase de origem: {self.sentence}"
        )

    def as_metadata(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "predicate": self.predicate,
            "subject": self.subject,
            "object": self.object,
            "arg2": self.arg2,
            "locations": list(self.locations),
            "times": list(self.times),
            "modifiers": list(self.modifiers),
            "sentence": self.sentence,
        }


def load_spacy_model(model_name: str = "pt_core_news_sm") -> spacy.Language:
    """Load the Portuguese spaCy model used by the symbolic extractor."""
    return spacy.load(model_name)


def _fact_from_frame(frame: dict[str, Any], event_id: str, sentence: str) -> SymbolicFact:
    argms = frame.get("ArgMs", {}) or {}
    return SymbolicFact(
        event_id=event_id,
        predicate=str(frame.get("predicado") or "evento"),
        subject=frame.get("Arg0"),
        object=frame.get("Arg1"),
        arg2=frame.get("Arg2"),
        locations=tuple(argms.get("loc", []) or []),
        times=tuple(argms.get("tmp", []) or []),
        modifiers=tuple(argms.get("outros", []) or []),
        sentence=sentence,
    )


def process_chapter_symbolically(chapter_text: str, nlp: spacy.Language) -> dict[str, Any]:
    """Run the first project's NLP pipeline over a full chapter.

    Returns a dictionary with cleaned text, sentences, graph elements,
    symbolic facts, and a NetworkX graph.
    """
    cleaned = limpar_texto(chapter_text)
    sentences = separar_frases(cleaned, nlp)

    elements: list[dict[str, Any]] = []
    facts: list[SymbolicFact] = []
    event_id = 0

    for sentence in sentences:
        doc = nlp(sentence.strip())
        frames = frames_module.extrair_todos_frames(doc)

        start_event_id = event_id
        new_elements, event_id = frames_module.frames_para_grafo_estruturado(frames, event_id)
        elements.extend(new_elements)

        for offset, frame in enumerate(frames):
            fact_event_id = f"{frame['predicado']}_{start_event_id + offset}"
            facts.append(_fact_from_frame(frame, fact_event_id, sentence))

    graph = grafo_module.construir_grafo(elements, nlp)

    return {
        "cleaned_text": cleaned,
        "sentences": sentences,
        "elements": elements,
        "facts": facts,
        "graph": graph,
    }


def symbolic_facts_to_documents(
    *,
    facts: list[SymbolicFact],
    chapter_id: str,
    chapter_title: str,
    language: str,
    created_at: str,
) -> list["Document"]:
    """Convert symbolic facts to documents that can be embedded and stored."""
    from langchain_core.documents import Document

    documents: list[Document] = []
    for index, fact in enumerate(facts, start=1):
        metadata = {
            "chapter_id": chapter_id,
            "chapter_title": chapter_title,
            "chunk_index": index,
            "language": language,
            "source": "symbolic_nlp_fact",
            "document_type": "symbolic_fact",
            "created_at": created_at,
            **fact.as_metadata(),
        }
        documents.append(Document(page_content=fact.as_text(), metadata=metadata))
    return documents


def facts_to_dataframe(facts: list[SymbolicFact]) -> pd.DataFrame:
    """Small table for Streamlit inspection."""
    return pd.DataFrame(
        [
            {
                "Evento": f.event_id,
                "Sujeito / Arg0": f.subject or "",
                "Relação": f.predicate,
                "Objeto / Arg1": f.object or "",
                "Local": ", ".join(f.locations),
                "Tempo": ", ".join(f.times),
                "Frase de origem": f.sentence,
            }
            for f in facts
        ]
    )


def elements_to_edges_dataframe(elements: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Origem": e["origem"],
                "Papel semântico": e["papel"],
                "Destino": e["destino"],
            }
            for e in elements
            if e.get("tipo") == "aresta"
        ]
    )


def elements_to_nodes_dataframe(elements: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Nó": e["id"],
                "Tipo do evento": e.get("attrs", {}).get("tipo_evento", ""),
            }
            for e in elements
            if e.get("tipo") == "no"
        ]
    )


def build_direct_fact_graph(facts: list[SymbolicFact]) -> nx.DiGraph:
    """Build a simpler concept-to-concept graph for symbolic QA/fact viewing."""
    graph = nx.DiGraph()
    for fact in facts:
        if fact.subject and fact.object:
            graph.add_node(fact.subject, tipo="Conceito")
            graph.add_node(fact.object, tipo="Conceito")
            graph.add_edge(
                fact.subject,
                fact.object,
                papel=fact.predicate,
                event_id=fact.event_id,
                sentence=fact.sentence,
            )
        elif fact.subject:
            graph.add_node(fact.subject, tipo="Conceito")
            graph.add_node(fact.event_id, tipo_evento=fact.predicate)
            graph.add_edge(fact.subject, fact.event_id, papel=fact.predicate)
        elif fact.object:
            graph.add_node(fact.event_id, tipo_evento=fact.predicate)
            graph.add_node(fact.object, tipo="Conceito")
            graph.add_edge(fact.event_id, fact.object, papel=fact.predicate)
    return graph


def answer_from_symbolic_facts(question: str, facts: list[SymbolicFact]) -> str:
    """Lightweight graph/fact answer for local fallback and debugging."""
    question_norm = normalizar_termo(question)
    if not facts:
        return "Ainda não há fatos simbólicos extraídos."

    subjects_and_objects = sorted(
        {term for fact in facts for term in (fact.subject, fact.object) if term},
        key=len,
        reverse=True,
    )
    matched = next((term for term in subjects_and_objects if term in question_norm), None)
    if not matched:
        return "Não encontrei nos fatos simbólicos um conceito mencionado nessa pergunta."

    candidate_facts = [
        fact for fact in facts if fact.subject == matched or fact.object == matched
    ]

    if "o que é" in question_norm or "defina" in question_norm:
        definitions = [
            fact.object for fact in candidate_facts
            if fact.subject == matched and fact.predicate == "instancia_de" and fact.object
        ]
        if definitions:
            return f"{matched.capitalize()} é {', '.join(definitions)}."

    if any(key in question_norm for key in ["como é", "característica", "caracteristicas", "propriedade"]):
        properties = [
            fact.object for fact in candidate_facts
            if fact.subject == matched and fact.predicate == "tem_propriedade" and fact.object
        ]
        if properties:
            return f"{matched.capitalize()} é {', '.join(properties)}."

    outgoing = [
        f"{fact.predicate} → {fact.object}"
        for fact in candidate_facts
        if fact.subject == matched and fact.object
    ]
    incoming = [
        f"{fact.subject} → {fact.predicate}"
        for fact in candidate_facts
        if fact.object == matched and fact.subject
    ]
    pieces = outgoing + incoming
    if pieces:
        return f"Relações simbólicas encontradas para '{matched}': " + "; ".join(pieces) + "."

    return "Encontrei o conceito, mas não há relações simbólicas suficientes para responder."

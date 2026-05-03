"""MongoDB Atlas Vector Search + Mistral RAG pipeline.

This module refactors the second student's single-file Streamlit app into
reusable functions and adds integration with the symbolic NLP pipeline.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from datetime import datetime, timezone
from typing import Any

import spacy
from langchain_core.documents import Document
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo.collection import Collection

from src.nlp_pipeline import (
    process_chapter_symbolically,
    symbolic_facts_to_documents,
)
from src.settings import AppSettings


DEFAULT_CHAPTER = """A mitose é um processo de divisão celular.
A mitose gera duas células-filhas geneticamente idênticas.
A mitose ocorre em células eucarióticas.
A mitose é importante para crescimento, regeneração e substituição celular.
A mitose é composta por prófase, metáfase, anáfase e telófase.
Na prófase, os cromossomos se condensam.
Na metáfase, os cromossomos se alinham no centro da célula.
Na anáfase, as cromátides-irmãs se separam.
Na telófase, formam-se dois novos núcleos.
O fuso mitótico ajuda a separar as cromátides-irmãs.
Os cromossomos carregam o material genético da célula.
"""


def clean_text(text: str) -> str:
    """Normalize pasted chapter text without changing the meaning."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def make_chapter_id(chapter_title: str, chapter_text: str) -> str:
    """Stable chapter id, useful when overwriting the same chapter."""
    raw = f"{chapter_title}\n{chapter_text}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def split_chapter_into_raw_documents(
    *,
    chapter_title: str,
    chapter_text: str,
    language: str,
    chapter_id: str,
    created_at: str,
    settings: AppSettings,
) -> list[Document]:
    """Split full chapter text into vector-searchable raw chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "],
    )

    documents: list[Document] = []
    chunks = splitter.split_text(chapter_text)
    for index, chunk in enumerate(chunks, start=1):
        chunk = chunk.strip()
        if not chunk:
            continue

        documents.append(
            Document(
                page_content=chunk,
                metadata={
                    "chapter_id": chapter_id,
                    "chapter_title": chapter_title,
                    "chunk_index": index,
                    "language": language,
                    "source": "user_pasted_chapter",
                    "document_type": "raw_chunk",
                    "created_at": created_at,
                },
            )
        )

    return documents


def build_ingestion_documents(
    *,
    chapter_title: str,
    chapter_text: str,
    language: str,
    chapter_id: str,
    settings: AppSettings,
    nlp: spacy.Language,
    include_raw_chunks: bool,
    include_symbolic_facts: bool,
) -> tuple[list[Document], dict[str, Any]]:
    """Create raw chunk documents and/or symbolic fact documents."""
    created_at = datetime.now(timezone.utc).isoformat()
    documents: list[Document] = []

    if include_raw_chunks:
        documents.extend(
            split_chapter_into_raw_documents(
                chapter_title=chapter_title,
                chapter_text=chapter_text,
                language=language,
                chapter_id=chapter_id,
                created_at=created_at,
                settings=settings,
            )
        )

    symbolic_result: dict[str, Any] = {
        "cleaned_text": chapter_text,
        "sentences": [],
        "elements": [],
        "facts": [],
        "graph": None,
    }
    if include_symbolic_facts:
        symbolic_result = process_chapter_symbolically(chapter_text, nlp)
        documents.extend(
            symbolic_facts_to_documents(
                facts=symbolic_result["facts"],
                chapter_id=chapter_id,
                chapter_title=chapter_title,
                language=language,
                created_at=created_at,
            )
        )

    return documents, symbolic_result


def save_chapter_to_mongodb(
    *,
    chapter_title: str,
    chapter_text: str,
    language: str,
    overwrite_existing: bool,
    include_raw_chunks: bool,
    include_symbolic_facts: bool,
    settings: AppSettings,
    nlp: spacy.Language,
    collection: Collection,
    vector_store: MongoDBAtlasVectorSearch,
) -> dict[str, Any]:
    """Clean, process, embed, and store a chapter in MongoDB.

    The integrated behavior is the key change from the original RAG app:
    raw chapter chunks and symbolic facts can be stored together in the same
    Atlas Vector Search collection. The LLM therefore receives both source
    text and extracted semantic triples/roles as grounding context.
    """
    cleaned = clean_text(chapter_text)
    if not cleaned:
        raise ValueError("Chapter text is empty.")
    if not include_raw_chunks and not include_symbolic_facts:
        raise ValueError("Select at least one ingestion source: raw chunks or symbolic facts.")

    chapter_id = make_chapter_id(chapter_title, cleaned)
    documents, symbolic_result = build_ingestion_documents(
        chapter_title=chapter_title,
        chapter_text=cleaned,
        language=language,
        chapter_id=chapter_id,
        settings=settings,
        nlp=nlp,
        include_raw_chunks=include_raw_chunks,
        include_symbolic_facts=include_symbolic_facts,
    )

    if not documents:
        raise ValueError("No valid documents were produced from this chapter.")

    if overwrite_existing:
        collection.delete_many({"chapter_id": chapter_id})

    inserted_ids = vector_store.add_documents(
        documents,
        ids=[str(uuid.uuid4()) for _ in documents],
    )

    raw_count = sum(1 for doc in documents if doc.metadata.get("document_type") == "raw_chunk")
    symbolic_count = sum(1 for doc in documents if doc.metadata.get("document_type") == "symbolic_fact")

    return {
        "chapter_id": chapter_id,
        "document_count": len(documents),
        "raw_chunk_count": raw_count,
        "symbolic_fact_count": symbolic_count,
        "inserted_count": len(inserted_ids),
        "symbolic_result": symbolic_result,
    }


def retrieve_relevant_documents(
    *,
    question: str,
    chapter_id: str | None,
    language: str | None,
    document_types: list[str] | None,
    settings: AppSettings,
    vector_store: MongoDBAtlasVectorSearch,
) -> list[tuple[Document, float]]:
    """Search MongoDB Atlas Vector Search for context similar to the question."""
    pre_filter: dict[str, Any] = {}
    if chapter_id:
        pre_filter["chapter_id"] = chapter_id
    if language:
        pre_filter["language"] = language
    if document_types:
        pre_filter["document_type"] = {"$in": document_types}

    try:
        results = vector_store.similarity_search_with_relevance_scores(
            question,
            k=settings.retrieval_k,
            pre_filter=pre_filter if pre_filter else None,
        )
    except TypeError:
        # Compatibility fallback for older langchain-mongodb versions.
        results = vector_store.similarity_search_with_relevance_scores(
            question,
            k=settings.retrieval_k,
        )
    except Exception:
        # Some Atlas/LangChain combinations are picky about pre_filter syntax.
        # Retry without server-side filtering and rely on the post-filter below.
        if not pre_filter:
            raise
        results = vector_store.similarity_search_with_relevance_scores(
            question,
            k=settings.retrieval_k,
        )

    filtered: list[tuple[Document, float]] = []
    for doc, score in results:
        metadata = doc.metadata or {}
        if score < settings.min_relevance_score:
            continue
        if chapter_id and metadata.get("chapter_id") != chapter_id:
            continue
        if language and metadata.get("language") != language:
            continue
        if document_types and metadata.get("document_type") not in document_types:
            continue
        filtered.append((doc, score))

    return filtered


def build_context(retrieved: list[tuple[Document, float]]) -> str:
    """Format retrieved documents before sending them to the LLM."""
    context_parts: list[str] = []
    for i, (doc, score) in enumerate(retrieved, start=1):
        metadata = doc.metadata or {}
        chapter_title = metadata.get("chapter_title", "Untitled chapter")
        chunk_index = metadata.get("chunk_index", "?")
        document_type = metadata.get("document_type", metadata.get("source", "unknown"))
        source = metadata.get("source", "unknown")
        context_parts.append(
            f"[Context {i} | type={document_type} | source={source} | "
            f"chapter={chapter_title} | index={chunk_index} | score={score:.3f}]\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(context_parts)


def answer_question_with_llm(
    *,
    question: str,
    retrieved: list[tuple[Document, float]],
    language: str,
    llm: Any,
) -> str:
    """Generate the final answer using only retrieved MongoDB context."""
    if not retrieved:
        return (
            "I could not find enough relevant information in MongoDB to answer this question."
            if language == "en"
            else "Não encontrei informações relevantes suficientes no MongoDB para responder a essa pergunta."
        )

    context = build_context(retrieved)

    if language == "pt":
        system_prompt = """
Você é um assistente educacional que responde perguntas usando exclusivamente o CONTEXTO fornecido.

O CONTEXTO pode conter dois tipos de informação:
- trechos originais do capítulo;
- fatos simbólicos/triplas extraídos por NLP, com papéis semânticos como Arg0, Arg1, local e tempo.

Regras obrigatórias:
1. Use somente informações explicitamente presentes no CONTEXTO.
2. Dê preferência a fatos simbólicos quando eles responderem diretamente à pergunta.
3. Use trechos originais para complementar a resposta ou resolver ambiguidades.
4. Não adicione explicações externas, mesmo que sejam corretas.
5. Se a resposta exigir informação que não está no CONTEXTO, diga: "O texto fornecido não traz essa informação."
6. Responda de forma clara, curta e fiel ao texto.
"""
        user_prompt = f"PERGUNTA:\n{question}\n\nCONTEXTO:\n{context}"
    else:
        system_prompt = """
You are an educational tutor. Answer only using the retrieved CONTEXT.
The CONTEXT may include raw chapter chunks and symbolic NLP facts/triples.
Prefer symbolic facts when they directly answer the question, and use raw chunks to clarify.
If the context does not contain the answer, say there is not enough information.
Be clear, concise, and grounded in the context.
"""
        user_prompt = f"QUESTION:\n{question}\n\nCONTEXT:\n{context}"

    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    return response.content

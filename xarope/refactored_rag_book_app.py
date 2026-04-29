"""
Streamlit RAG app for book chapters.

Pipeline:
1. User pastes a book chapter into Streamlit.
2. The chapter is cleaned and split into chunks.
3. Chunks are embedded and stored in MongoDB Atlas Vector Search.
4. User asks a question.
5. The app retrieves relevant chunks from MongoDB.
6. The retrieved context is sent to Mistral.
7. Mistral returns an answer grounded in the stored chapter.

Required environment variables:
- MONGODB_URI
- MISTRAL_API_KEY
- HF_TOKEN

Recommended .env example:
MONGODB_URI=mongodb+srv://...
MISTRAL_API_KEY=...
HF_TOKEN=...
MONGODB_DATABASE=book_rag_db
MONGODB_COLLECTION=chapters
MONGODB_VECTOR_INDEX=vector_index
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
CHAT_MODEL=mistral-small-latest
"""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo import MongoClient
from pymongo.collection import Collection


# ============================================================
# 1. PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Book Chapter RAG with MongoDB",
    page_icon="📚",
    layout="wide",
)


# ============================================================
# 2. ENVIRONMENT AND CONSTANTS
# ============================================================

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "book_rag_db")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "chapters")
MONGODB_VECTOR_INDEX = os.getenv("MONGODB_VECTOR_INDEX", "vector_index")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
CHAT_MODEL = os.getenv("CHAT_MODEL", "mistral-small-latest")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))
MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.65"))

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


# ============================================================
# 3. VALIDATION
# ============================================================

def validate_environment() -> None:
    missing = []
    if not MONGODB_URI:
        missing.append("MONGODB_URI")
    if not MISTRAL_API_KEY:
        missing.append("MISTRAL_API_KEY")
    if not HF_TOKEN:
        missing.append("HF_TOKEN")

    if missing:
        st.error(
            "Missing environment variables: " + ", ".join(missing) +
            ". Add them to your .env file or deployment secrets."
        )
        st.stop()


# ============================================================
# 4. CACHED CLIENTS
# ============================================================

@st.cache_resource(show_spinner=False)
def get_mongo_collection() -> Collection:
    validate_environment()
    client = MongoClient(MONGODB_URI)
    return client[MONGODB_DATABASE][MONGODB_COLLECTION]


@st.cache_resource(show_spinner=False)
def get_embeddings() -> HuggingFaceEndpointEmbeddings:
    validate_environment()
    return HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token=HF_TOKEN,
        model=EMBEDDING_MODEL,
        task="feature-extraction",
    )


@st.cache_resource(show_spinner=False)
def get_llm() -> ChatMistralAI:
    validate_environment()
    return ChatMistralAI(
        model=CHAT_MODEL,
        api_key=MISTRAL_API_KEY,
        temperature=0.2,
    )


@st.cache_resource(show_spinner=False)
def get_vector_store() -> MongoDBAtlasVectorSearch:
    collection = get_mongo_collection()
    embeddings = get_embeddings()
    return MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=MONGODB_VECTOR_INDEX,
    )


# ============================================================
# 5. TEXT PROCESSING
# ============================================================

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


def split_chapter_into_documents(
    *,
    chapter_title: str,
    chapter_text: str,
    language: str,
    chapter_id: str,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "],
    )

    chunks = splitter.split_text(chapter_text)
    created_at = datetime.now(timezone.utc).isoformat()

    documents = []
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
                    "created_at": created_at,
                },
            )
        )

    return documents


# ============================================================
# 6. MONGODB WRITE PIPELINE
# ============================================================

def save_chapter_to_mongodb(
    *,
    chapter_title: str,
    chapter_text: str,
    language: str,
    overwrite_existing: bool,
) -> dict[str, Any]:
    """
    Converts a chapter into vector-searchable chunks and stores them in MongoDB.
    """
    cleaned = clean_text(chapter_text)
    if not cleaned:
        raise ValueError("Chapter text is empty.")

    chapter_id = make_chapter_id(chapter_title, cleaned)
    documents = split_chapter_into_documents(
        chapter_title=chapter_title,
        chapter_text=cleaned,
        language=language,
        chapter_id=chapter_id,
    )

    if not documents:
        raise ValueError("No valid chunks were produced from this chapter.")

    collection = get_mongo_collection()
    vector_store = get_vector_store()

    if overwrite_existing:
        collection.delete_many({"chapter_id": chapter_id})

    inserted_ids = vector_store.add_documents(documents, ids=[str(uuid.uuid4()) for _ in documents])

    return {
        "chapter_id": chapter_id,
        "chunk_count": len(documents),
        "inserted_count": len(inserted_ids),
    }


# ============================================================
# 7. RETRIEVAL + LLM ANSWERING PIPELINE
# ============================================================

def retrieve_relevant_chunks(
    *,
    question: str,
    chapter_id: str | None,
    language: str | None,
    k: int = RETRIEVAL_K,
) -> list[tuple[Document, float]]:
    """Search MongoDB Atlas Vector Search for chunks similar to the question."""
    vector_store = get_vector_store()

    # langchain-mongodb accepts a MongoDB pre-filter in many versions.
    # If your installed version does not support pre_filter, remove the argument
    # and rely on the Python post-filter below.
    pre_filter: dict[str, Any] = {}
    if chapter_id:
        pre_filter["chapter_id"] = chapter_id
    if language:
        pre_filter["language"] = language

    try:
        results = vector_store.similarity_search_with_relevance_scores(
            question,
            k=k,
            pre_filter=pre_filter if pre_filter else None,
        )
    except TypeError:
        # Compatibility fallback for older langchain-mongodb versions.
        results = vector_store.similarity_search_with_relevance_scores(question, k=k)

    filtered: list[tuple[Document, float]] = []
    for doc, score in results:
        metadata = doc.metadata or {}

        if score < MIN_RELEVANCE_SCORE:
            continue
        if chapter_id and metadata.get("chapter_id") != chapter_id:
            continue
        if language and metadata.get("language") != language:
            continue

        filtered.append((doc, score))

    return filtered


def build_context(retrieved: list[tuple[Document, float]]) -> str:
    context_parts = []
    for i, (doc, score) in enumerate(retrieved, start=1):
        metadata = doc.metadata or {}
        chapter_title = metadata.get("chapter_title", "Untitled chapter")
        chunk_index = metadata.get("chunk_index", "?")
        context_parts.append(
            f"[Context {i} | chapter={chapter_title} | chunk={chunk_index} | score={score:.3f}]\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(context_parts)


def answer_question_with_llm(
    *,
    question: str,
    retrieved: list[tuple[Document, float]],
    language: str,
) -> str:
    if not retrieved:
        return (
            "I could not find enough relevant information in MongoDB to answer this question."
            if language == "en"
            else "Não encontrei informações relevantes suficientes no MongoDB para responder a essa pergunta."
        )

    context = build_context(retrieved)
    llm = get_llm()

    if language == "pt":
        system_prompt = (
            """
            Você é um assistente educacional que responde perguntas usando exclusivamente o CONTEXTO fornecido.

            Regras obrigatórias:
            1. Use somente informações explicitamente presentes no CONTEXTO.
            2. Não adicione explicações externas, mesmo que sejam corretas.
            3. Não use conhecimento geral que não esteja no CONTEXTO.
            4. Se a resposta exigir informação que não está no CONTEXTO, diga:
            "O texto fornecido não traz essa informação."
            5. Responda de forma clara, curta e fiel ao texto.
            6. Não explique termos que o CONTEXTO não explicou.
            """
        )
        user_prompt = f"PERGUNTA:\n{question}\n\nCONTEXTO:\n{context}"
    else:
        system_prompt = (
            "You are an educational tutor. Answer only using the CONTEXT retrieved from MongoDB. "
            "If the context does not contain the answer, say there is not enough information. "
            "Be clear, concise, and didactic."
        )
        user_prompt = f"QUESTION:\n{question}\n\nCONTEXT:\n{context}"

    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    return response.content


# ============================================================
# 8. STREAMLIT UI
# ============================================================

st.title("📚 Interactive Book Chapter RAG")
st.caption("Text input → MongoDB Atlas Vector Search → retrieved context → Mistral answer")

with st.sidebar:
    st.header("⚙️ Settings")
    language = st.selectbox("Language", options=["pt", "en"], index=0)
    overwrite_existing = st.checkbox("Overwrite same chapter if already stored", value=True)

    st.markdown("---")
    st.write("**MongoDB target**")
    st.code(f"{MONGODB_DATABASE}.{MONGODB_COLLECTION}")
    st.write("**Vector index**")
    st.code(MONGODB_VECTOR_INDEX)

    st.markdown("---")
    st.write("**Retrieval**")
    st.write(f"Top K: `{RETRIEVAL_K}`")
    st.write(f"Minimum score: `{MIN_RELEVANCE_SCORE}`")


if "active_chapter_id" not in st.session_state:
    st.session_state.active_chapter_id = None
if "active_chapter_title" not in st.session_state:
    st.session_state.active_chapter_title = None

chapter_tab, chat_tab, inspect_tab = st.tabs([
    "1️⃣ Store chapter",
    "2️⃣ Ask questions",
    "3️⃣ Inspect retrieval",
])


with chapter_tab:
    st.header("1️⃣ Paste a book chapter")

    chapter_title = st.text_input(
        "Chapter title",
        value="Mitose",
        placeholder="Example: Chapter 3 — Cell Division",
    )

    chapter_text = st.text_area(
        "Chapter text",
        value=DEFAULT_CHAPTER,
        height=360,
        placeholder="Paste the full chapter here...",
    )

    if st.button("📤 Process and store in MongoDB", type="primary"):
        try:
            with st.spinner("Creating chunks, embeddings, and saving to MongoDB..."):
                result = save_chapter_to_mongodb(
                    chapter_title=chapter_title.strip() or "Untitled chapter",
                    chapter_text=chapter_text,
                    language=language,
                    overwrite_existing=overwrite_existing,
                )

            st.session_state.active_chapter_id = result["chapter_id"]
            st.session_state.active_chapter_title = chapter_title.strip() or "Untitled chapter"

            st.success(
                f"Chapter stored successfully. "
                f"Chapter ID: `{result['chapter_id']}`. "
                f"Chunks inserted: `{result['inserted_count']}`."
            )
        except Exception as exc:
            st.error(f"Could not store chapter: {exc}")


with chat_tab:
    st.header("2️⃣ Ask a question about the stored chapter")

    if st.session_state.active_chapter_id:
        st.info(
            f"Active chapter: **{st.session_state.active_chapter_title}** "
            f"(`{st.session_state.active_chapter_id}`)"
        )
    else:
        st.warning("Store a chapter first, or manually enter a chapter ID below.")

    manual_chapter_id = st.text_input(
        "Chapter ID to search",
        value=st.session_state.active_chapter_id or "",
        placeholder="Paste a chapter_id, or store a chapter first.",
    )

    question = st.text_input(
        "Your question",
        placeholder="Example: O que é mitose?",
    )

    if st.button("💬 Retrieve and answer", type="primary"):
        if not question.strip():
            st.warning("Please type a question first.")
        elif not manual_chapter_id.strip():
            st.warning("Please store a chapter or enter a chapter ID first.")
        else:
            try:
                with st.spinner("Searching MongoDB and asking the LLM..."):
                    retrieved = retrieve_relevant_chunks(
                        question=question.strip(),
                        chapter_id=manual_chapter_id.strip(),
                        language=language,
                    )
                    answer = answer_question_with_llm(
                        question=question.strip(),
                        retrieved=retrieved,
                        language=language,
                    )

                st.subheader("Answer")
                st.write(answer)

                st.session_state.last_retrieved = retrieved
            except Exception as exc:
                st.error(f"Could not answer question: {exc}")


with inspect_tab:
    st.header("3️⃣ Retrieved chunks")
    st.caption("This shows what MongoDB returned before the LLM generated the final answer.")

    retrieved = st.session_state.get("last_retrieved", [])
    if not retrieved:
        st.info("Ask a question first to inspect retrieved chunks.")
    else:
        for i, (doc, score) in enumerate(retrieved, start=1):
            metadata = doc.metadata or {}
            with st.expander(f"Chunk {i} — score {score:.3f}"):
                st.write("**Metadata**")
                st.json(metadata)
                st.write("**Text**")
                st.write(doc.page_content)


st.markdown("---")
st.markdown(
    """
### How this refactored app works

1. You paste a chapter.
2. The app cleans the text.
3. The app splits the chapter into overlapping chunks.
4. Hugging Face creates an embedding for each chunk.
5. MongoDB stores each chunk, its metadata, and its embedding.
6. You ask a question.
7. MongoDB Atlas Vector Search retrieves the most relevant chunks.
8. Mistral receives the question plus retrieved context.
9. The app displays the LLM answer and lets you inspect the retrieved chunks.

### MongoDB Atlas Vector Search requirement

Create a vector search index on the collection configured above. The default LangChain fields are usually:

- text field: `text`
- embedding field: `embedding`
- dimensions: depends on the selected embedding model
- similarity: cosine

If your collection uses different field names, configure `MongoDBAtlasVectorSearch` with the matching `text_key` and `embedding_key`.
"""
)

from __future__ import annotations

import pandas as pd
import spacy
import streamlit as st

from src import grafo as grafo_module
from src.nlp_pipeline import (
    answer_from_symbolic_facts,
    build_direct_fact_graph,
    elements_to_edges_dataframe,
    elements_to_nodes_dataframe,
    facts_to_dataframe,
    process_chapter_symbolically,
)
from src.rag_pipeline import (
    DEFAULT_CHAPTER,
    answer_question_with_llm,
    clean_text,
    make_chapter_id,
    retrieve_relevant_documents,
    save_chapter_to_mongodb,
)
from src.services import (
    build_embeddings,
    build_llm,
    build_mongo_collection,
    build_vector_store,
)
from src.settings import get_settings, missing_required_env


st.set_page_config(
    page_title="Livro Didático Interativo — NLP + RAG",
    page_icon="📚",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_spacy_model() -> spacy.Language:
    try:
        return spacy.load("pt_core_news_sm")
    except OSError as exc:
        raise RuntimeError(
            "Modelo spaCy pt_core_news_sm não encontrado. "
            "Instale com `python -m spacy download pt_core_news_sm` "
            "ou use `uv sync` com o pyproject deste projeto."
        ) from exc


@st.cache_resource(show_spinner=False)
def get_cached_mongo_collection():
    return build_mongo_collection(get_settings())


@st.cache_resource(show_spinner=False)
def get_cached_embeddings():
    return build_embeddings(get_settings())


@st.cache_resource(show_spinner=False)
def get_cached_vector_store():
    settings = get_settings()
    return build_vector_store(
        collection=get_cached_mongo_collection(),
        embeddings=get_cached_embeddings(),
        settings=settings,
    )


@st.cache_resource(show_spinner=False)
def get_cached_llm():
    return build_llm(get_settings())


def ensure_session_state() -> None:
    defaults = {
        "active_chapter_id": None,
        "active_chapter_title": None,
        "last_symbolic_result": None,
        "last_retrieved": [],
        "last_answer": None,
        "last_symbolic_answer": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def show_environment_status(settings) -> None:
    missing = missing_required_env(settings)
    if missing:
        st.warning(
            "MongoDB/RAG mode is not ready. Missing: " + ", ".join(missing)
        )
    else:
        st.success("MongoDB/RAG environment variables are configured.")


def render_symbolic_result(symbolic_result: dict | None) -> None:
    if not symbolic_result:
        st.info("Process a chapter first to see symbolic NLP output.")
        return

    facts = symbolic_result.get("facts", [])
    elements = symbolic_result.get("elements", [])
    graph = symbolic_result.get("graph")

    st.subheader("Symbolic NLP summary")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Sentences", len(symbolic_result.get("sentences", [])))
    col_b.metric("Symbolic facts", len(facts))
    col_c.metric("Graph edges", graph.number_of_edges() if graph else 0)

    graph_tab, facts_tab, edges_tab, nodes_tab = st.tabs(
        ["🕸️ Event graph", "📌 Symbolic facts", "➡️ Event-role edges", "⭕ Nodes"]
    )

    with graph_tab:
        if graph is not None and graph.number_of_nodes() > 0:
            st.pyplot(grafo_module.desenhar_grafo(graph))
        else:
            st.info("No graph nodes were produced.")

    with facts_tab:
        if facts:
            st.dataframe(facts_to_dataframe(facts), use_container_width=True)
            st.markdown("#### Text form used for RAG")
            for fact in facts:
                st.code(fact.as_text())
        else:
            st.info("No symbolic facts were extracted.")

    with edges_tab:
        df = elements_to_edges_dataframe(elements)
        if df.empty:
            st.info("No event-role edges were extracted.")
        else:
            st.dataframe(df, use_container_width=True)

    with nodes_tab:
        df = elements_to_nodes_dataframe(elements)
        if df.empty:
            st.info("No event nodes were extracted.")
        else:
            st.dataframe(df, use_container_width=True)


ensure_session_state()
settings = get_settings()

st.title("📚 Livro Didático Interativo — NLP simbólico + MongoDB RAG")
st.caption(
    "Capítulo → spaCy frames/triplas → grafo semântico + documentos vetoriais → "
    "MongoDB Atlas Vector Search → Mistral"
)

with st.sidebar:
    st.header("⚙️ Settings")
    language = st.selectbox("Language", options=["pt", "en"], index=0)
    overwrite_existing = st.checkbox("Overwrite same chapter if already stored", value=True)

    st.markdown("---")
    st.write("**What to store in MongoDB**")
    include_raw_chunks = st.checkbox("Raw chapter chunks", value=True)
    include_symbolic_facts = st.checkbox("Symbolic NLP facts/triples", value=True)

    st.markdown("---")
    st.write("**What to retrieve for the LLM**")
    retrieve_raw = st.checkbox("Retrieve raw chunks", value=True)
    retrieve_symbolic = st.checkbox("Retrieve symbolic facts", value=True)

    st.markdown("---")
    st.write("**MongoDB target**")
    st.code(f"{settings.mongodb_database}.{settings.mongodb_collection}")
    st.write("**Vector index**")
    st.code(settings.mongodb_vector_index)

    st.markdown("---")
    st.write("**Models**")
    st.write(f"Embedding: `{settings.embedding_model}`")
    st.write(f"Chat: `{settings.chat_model}`")

    st.markdown("---")
    st.write("**Retrieval**")
    st.write(f"Top K: `{settings.retrieval_k}`")
    st.write(f"Minimum score: `{settings.min_relevance_score}`")

    st.markdown("---")
    show_environment_status(settings)

chapter_tab, chat_tab, inspect_tab, symbolic_tab = st.tabs(
    [
        "1️⃣ Store/process chapter",
        "2️⃣ Ask questions",
        "3️⃣ Inspect retrieval",
        "4️⃣ Symbolic graph",
    ]
)

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

    col_local, col_store = st.columns(2)

    with col_local:
        if st.button("🔎 Process symbolic NLP locally"):
            try:
                with st.spinner("Running spaCy frame extraction and graph construction..."):
                    nlp = get_spacy_model()
                    symbolic_result = process_chapter_symbolically(chapter_text, nlp)
                    cleaned = clean_text(chapter_text)
                    chapter_id = make_chapter_id(chapter_title.strip() or "Untitled chapter", cleaned)

                st.session_state.active_chapter_id = chapter_id
                st.session_state.active_chapter_title = chapter_title.strip() or "Untitled chapter"
                st.session_state.last_symbolic_result = symbolic_result
                st.success(
                    f"Local symbolic processing complete. Chapter ID: `{chapter_id}`. "
                    f"Facts extracted: `{len(symbolic_result['facts'])}`."
                )
            except Exception as exc:
                st.error(f"Could not run symbolic NLP processing: {exc}")

    with col_store:
        if st.button("📤 Process and store in MongoDB", type="primary"):
            missing = missing_required_env(settings)
            if missing:
                st.error("Cannot store in MongoDB. Missing: " + ", ".join(missing))
            else:
                try:
                    with st.spinner("Creating chunks, symbolic facts, embeddings, and saving to MongoDB..."):
                        nlp = get_spacy_model()
                        result = save_chapter_to_mongodb(
                            chapter_title=chapter_title.strip() or "Untitled chapter",
                            chapter_text=chapter_text,
                            language=language,
                            overwrite_existing=overwrite_existing,
                            include_raw_chunks=include_raw_chunks,
                            include_symbolic_facts=include_symbolic_facts,
                            settings=settings,
                            nlp=nlp,
                            collection=get_cached_mongo_collection(),
                            vector_store=get_cached_vector_store(),
                        )

                    st.session_state.active_chapter_id = result["chapter_id"]
                    st.session_state.active_chapter_title = chapter_title.strip() or "Untitled chapter"
                    st.session_state.last_symbolic_result = result["symbolic_result"]
                    st.success(
                        f"Chapter stored successfully. Chapter ID: `{result['chapter_id']}`. "
                        f"Inserted: `{result['inserted_count']}` documents "
                        f"({result['raw_chunk_count']} raw chunks, "
                        f"{result['symbolic_fact_count']} symbolic facts)."
                    )
                except Exception as exc:
                    st.error(f"Could not store chapter: {exc}")

    st.markdown("---")
    render_symbolic_result(st.session_state.last_symbolic_result)

with chat_tab:
    st.header("2️⃣ Ask a question about the stored chapter")

    if st.session_state.active_chapter_id:
        st.info(
            f"Active chapter: **{st.session_state.active_chapter_title}** "
            f"(`{st.session_state.active_chapter_id}`)"
        )
    else:
        st.warning("Store/process a chapter first, or manually enter a chapter ID below.")

    manual_chapter_id = st.text_input(
        "Chapter ID to search",
        value=st.session_state.active_chapter_id or "",
        placeholder="Paste a chapter_id, or process/store a chapter first.",
    )

    question = st.text_input(
        "Your question",
        placeholder="Example: O que é mitose?",
    )

    if st.button("💬 Retrieve and answer", type="primary"):
        if not question.strip():
            st.warning("Please type a question first.")
        elif not manual_chapter_id.strip():
            st.warning("Please store/process a chapter or enter a chapter ID first.")
        else:
            missing = missing_required_env(settings)
            if missing:
                st.error("Cannot use MongoDB/RAG mode. Missing: " + ", ".join(missing))
            else:
                try:
                    document_types = []
                    if retrieve_raw:
                        document_types.append("raw_chunk")
                    if retrieve_symbolic:
                        document_types.append("symbolic_fact")

                    if not document_types:
                        st.warning("Select at least one retrieval source in the sidebar.")
                    else:
                        with st.spinner("Searching MongoDB and asking Mistral..."):
                            retrieved = retrieve_relevant_documents(
                                question=question.strip(),
                                chapter_id=manual_chapter_id.strip(),
                                language=language,
                                document_types=document_types,
                                settings=settings,
                                vector_store=get_cached_vector_store(),
                            )
                            answer = answer_question_with_llm(
                                question=question.strip(),
                                retrieved=retrieved,
                                language=language,
                                llm=get_cached_llm(),
                            )

                        st.session_state.last_retrieved = retrieved
                        st.session_state.last_answer = answer
                        st.subheader("Mistral RAG answer")
                        st.write(answer)
                except Exception as exc:
                    st.error(f"Could not answer question: {exc}")

    st.markdown("---")
    st.subheader("Local symbolic answer/debug fallback")
    if st.button("🧠 Answer using symbolic facts only"):
        symbolic_result = st.session_state.last_symbolic_result
        if not question.strip():
            st.warning("Please type a question first.")
        elif not symbolic_result:
            st.warning("Process the chapter locally first so symbolic facts are available.")
        else:
            facts = symbolic_result.get("facts", [])
            symbolic_answer = answer_from_symbolic_facts(question.strip(), facts)
            st.session_state.last_symbolic_answer = symbolic_answer
            st.write(symbolic_answer)

    if st.session_state.last_symbolic_answer:
        st.caption("Last symbolic-only answer")
        st.info(st.session_state.last_symbolic_answer)

with inspect_tab:
    st.header("3️⃣ Retrieved MongoDB context")
    st.caption("This shows what MongoDB returned before Mistral generated the final answer.")

    retrieved = st.session_state.get("last_retrieved", [])
    if not retrieved:
        st.info("Ask a RAG question first to inspect retrieved documents.")
    else:
        summary_rows = []
        for i, (doc, score) in enumerate(retrieved, start=1):
            metadata = doc.metadata or {}
            summary_rows.append(
                {
                    "Rank": i,
                    "Score": round(score, 3),
                    "Type": metadata.get("document_type"),
                    "Source": metadata.get("source"),
                    "Index": metadata.get("chunk_index"),
                    "Predicate": metadata.get("predicate", ""),
                    "Subject": metadata.get("subject", ""),
                    "Object": metadata.get("object", ""),
                }
            )
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

        for i, (doc, score) in enumerate(retrieved, start=1):
            metadata = doc.metadata or {}
            label = (
                f"Context {i} — {metadata.get('document_type', 'unknown')} "
                f"— score {score:.3f}"
            )
            with st.expander(label):
                st.write("**Metadata**")
                st.json(metadata)
                st.write("**Text sent as context**")
                st.write(doc.page_content)

with symbolic_tab:
    st.header("4️⃣ Symbolic graph and extracted facts")
    render_symbolic_result(st.session_state.last_symbolic_result)

    symbolic_result = st.session_state.last_symbolic_result
    if symbolic_result and symbolic_result.get("facts"):
        st.markdown("---")
        st.subheader("Direct fact graph")
        direct_graph = build_direct_fact_graph(symbolic_result["facts"])
        if direct_graph.number_of_nodes() > 0:
            st.pyplot(grafo_module.desenhar_grafo(direct_graph))

st.markdown("---")
st.markdown(
    """
### Integrated pipeline

1. Paste a chapter.
2. The app runs the original symbolic NLP pipeline: spaCy sentence parsing → semantic frames → graph elements → symbolic facts/triples.
3. The app also splits the raw chapter into overlapping chunks.
4. Raw chunks and symbolic facts are embedded with Hugging Face.
5. MongoDB Atlas Vector Search stores both kinds of documents.
6. A user question retrieves relevant raw chunks and/or symbolic facts.
7. Mistral receives the retrieved context and generates a grounded educational answer.
8. The Inspect tab shows exactly what was retrieved before the LLM answered.
"""
)

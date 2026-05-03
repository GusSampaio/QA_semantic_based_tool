# QA Semantic RAG Integrated Project

This project integrates two parts of the original school project:

1. **Symbolic NLP / structured text processing**
   - Uses spaCy Portuguese parsing.
   - Extracts semantic frames, roles, and graph elements.
   - Builds a NetworkX semantic/event graph.
   - Produces symbolic facts/triples that can be inspected and reused.

2. **Streamlit + MongoDB + Hugging Face + Mistral RAG**
   - Lets the user paste a book chapter.
   - Stores raw chapter chunks and symbolic NLP facts in MongoDB Atlas Vector Search.
   - Retrieves relevant context for a user question.
   - Sends retrieved context to Mistral for a grounded final answer.

The important integration is that the app no longer stores only raw text chunks. It can also store the symbolic NLP facts extracted from the first project as vector-searchable documents. During RAG, MongoDB can retrieve raw chapter chunks, symbolic facts, or both.

---

## Final flow

```text
Book chapter
  → clean text
  → spaCy symbolic NLP processing
  → semantic frames / roles / graph elements
  → symbolic facts/triples
  → raw chunks + symbolic fact documents
  → Hugging Face embeddings
  → MongoDB Atlas Vector Search
  → user question
  → relevant raw chunks + symbolic facts
  → Mistral prompt
  → final answer
```

---

## File structure

```text
QA_semantic_rag_integrated/
├── app.py                         # Main integrated Streamlit app
├── pyproject.toml                 # uv-compatible project dependencies
├── requirements.txt               # pip-compatible dependency list
├── .env.example                   # Example environment variables
├── .python-version
├── .gitignore
├── README.md
├── doc/
│   └── regras_utlizadas.txt       # Original notes explaining symbolic extraction rules
├── src/
│   ├── __init__.py
│   ├── auxiliares.py              # Original text helpers: clean, split sentences, normalize terms
│   ├── extracoes.py               # Original extraction wrapper
│   ├── frames.py                  # Original semantic-frame extraction rules
│   ├── grafo.py                   # Original NetworkX graph construction / visualization / QA helper
│   ├── nlp_pipeline.py            # New integration layer for symbolic facts and graph inspection
│   ├── rag_pipeline.py            # New refactored MongoDB/RAG ingestion and answering pipeline
│   ├── services.py                # MongoDB, Hugging Face, and Mistral client factories
│   └── settings.py                # Environment-backed app configuration
└── tests/
    └── test_symbolic_fact.py      # Lightweight test for fact rendering
```

---

## Setup with uv

```bash
uv sync
cp .env.example .env
# Edit .env with your real credentials
uv run streamlit run app.py
```

## Setup with pip

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your real credentials
streamlit run app.py
```

If the spaCy model is not installed, install it manually:

```bash
python -m spacy download pt_core_news_sm
```

---

## Required environment variables

Required for MongoDB/RAG mode:

```env
MONGODB_URI=mongodb+srv://...
MISTRAL_API_KEY=...
HF_TOKEN=...
```

Optional:

```env
MONGODB_DATABASE=book_rag_db
MONGODB_COLLECTION=chapters
MONGODB_VECTOR_INDEX=vector_index
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
CHAT_MODEL=mistral-small-latest
CHUNK_SIZE=700
CHUNK_OVERLAP=120
RETRIEVAL_K=8
MIN_RELEVANCE_SCORE=0.55
```

---

## MongoDB Atlas Vector Search index

Create an Atlas Vector Search index on the configured collection. LangChain's default field names are usually:

- text field: `text`
- embedding field: `embedding`

The vector dimensions must match the selected embedding model. For the default MiniLM model, check the Hugging Face model card and configure Atlas accordingly.

A typical index shape is:

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 384,
      "similarity": "cosine"
    },
    { "type": "filter", "path": "chapter_id" },
    { "type": "filter", "path": "language" },
    { "type": "filter", "path": "document_type" }
  ]
}
```

If your LangChain/MongoDB version uses different field names, configure `MongoDBAtlasVectorSearch` in `src/services.py` with matching `text_key` and `embedding_key`.

---

## How to use the app

1. Open the app with `streamlit run app.py`.
2. Paste a book chapter in tab **1️⃣ Store/process chapter**.
3. Click **Process symbolic NLP locally** to inspect extracted facts and graphs without MongoDB.
4. Click **Process and store in MongoDB** to embed and store raw chunks and/or symbolic facts.
5. Go to **2️⃣ Ask questions** and ask a question about the active chapter.
6. Inspect the retrieved context in **3️⃣ Inspect retrieval**.
7. Inspect extracted symbolic output in **4️⃣ Symbolic graph**.

---

## Important design choice

The first project produces an event-role graph, for example:

```text
gerar_0 --Arg0--> mitose
gerar_0 --Arg1--> células-filhas
```

For RAG, this project also renders each frame as a symbolic fact document:

```text
Fato simbólico extraído por NLP: mitose --[gerar]--> células-filhas
Papéis semânticos: Arg0/agente/sujeito=mitose; Arg1/paciente/objeto=células-filhas
Frase de origem: A mitose gera células-filhas.
```

This makes the symbolic extraction useful for vector retrieval and LLM grounding.

---

## Limitations

- The symbolic extractor is rule-based and depends on spaCy dependency parses, so extraction quality depends on sentence structure and the Portuguese model output.
- MongoDB, Hugging Face, and Mistral calls require real credentials and network access.
- I could run static syntax checks in this environment, but I could not fully test Atlas Vector Search or Mistral without API keys and a configured MongoDB Atlas index.

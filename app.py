import streamlit as st
import spacy
import pandas as pd

from src.auxiliares import limpar_texto, separar_frases
from src.extracoes import extrair_triplas_frames, extrair_triplas_frames_com_metodo
import src.grafo as grafo_module
from src.llm import LLM
from src.settings import AppSettings

# Configuracao geral da pagina
st.set_page_config(
    page_title="Livro Didático Interativo com Grafo Semântico",
    page_icon="📖",
    layout="wide"
)

# Carregando modelo spacy para portugues
@st.cache_resource
def carregar_modelo_spacy():
    try:
        return spacy.load("pt_core_news_sm")
    except OSError:
        st.error(
            "Modelo pt_core_news_sm não encontrado. "
            "Modelo pt_core_news_sm não encontrado. Execute: uv run python -m spacy download pt_core_news_sm"
        )
        st.stop()

nlp = carregar_modelo_spacy()

# Texto de exemplo
TEXTO_PADRAO = """A mitose gera células-filhas.
As células-filhas foram geradas pela mitose.
A mitose ocorre em células eucariontes em 2020.
A mitose ocorre em células eucariontes e produz células-filhas.
As células que foram geradas pela mitose entram em divisão.
As células que foram geradas pela mitose e organizadas pelo núcleo entram em divisão."""

# Inicialização do estado -------------------------------------------------------
if "triplas" not in st.session_state:
    frases_iniciais = separar_frases(limpar_texto(TEXTO_PADRAO), nlp)
    triplas_iniciais = extrair_triplas_frames(frases_iniciais, nlp)
    grafo_inicial = grafo_module.construir_grafo(triplas_iniciais, nlp)

    st.session_state.frases = frases_iniciais
    st.session_state.triplas = triplas_iniciais
    st.session_state.grafo = grafo_inicial
    st.session_state.metodo_atual = "Simbólico"

# Sidebar — seleção de método ---------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configurações")
    metodo_selecionado = st.radio(
        "Método de extração:",
        options=["Simbólico", "Estatístico"],
        index=0,
        help=(
            "**Simbólico:** regras linguísticas determinísticas sobre labels "
            "Universal Dependencies (UD). Alta precisão, sem gradação de confiança.\n\n"
            "**Estatístico:** candidatos a argumento pontuados por frequência de "
            "termos no corpus e distância na árvore de dependências. Inclui fallback "
            "por chunks nominais quando sinais diretos estão ausentes."
        ),
    )
    metodo_key = "estatistico" if metodo_selecionado == "Estatístico" else "simbolico"

# UI ----------------------------------------------------------------------------
st.title("📖 Livro Didático Virtual Interativo")

st.markdown("""
Pipeline: **texto → segmentação → árvore UD → extração → grafo**
""")

col1, col2 = st.columns([1, 1])

# COLUNA TEXTO ------------------------------------------------------------------
with col1:
    st.header("📖 Capítulo")
    texto = st.text_area("Cole ou edite o capítulo:", value=TEXTO_PADRAO, height=370)

    if st.button("🔎 Processar texto e gerar grafo"):
        frases = separar_frases(limpar_texto(texto), nlp)
        triplas = extrair_triplas_frames_com_metodo(frases, nlp, metodo=metodo_key)
        grafo = grafo_module.construir_grafo(triplas, nlp)

        st.session_state.frases = frases
        st.session_state.triplas = triplas
        st.session_state.grafo = grafo
        st.session_state.metodo_atual = metodo_selecionado

        st.success(
            f"Texto processado com método **{metodo_selecionado}**. "
            f"Foram extraídas {len(triplas)} triplas e o grafo resultante tem "
            f"{grafo.number_of_nodes()} nós e {grafo.number_of_edges()} arestas."
        )

# COLUNA PERGUNTAS --------------------------------------------------------------
with col2:
    st.header("💬 Pergunte ao grafo")
    pergunta = st.text_input("Digite uma pergunta:", placeholder="O que é mitose?")

    if st.button("Responder"):
        if pergunta.strip():

            # tuplas_grafo = st.session_state.triplas
            resposta_grafo = grafo_module.responder_pergunta(st.session_state.grafo)
            LLM_model = LLM(AppSettings())
            resposta_llm = LLM_model.answer_question_with_llm(
                question=pergunta,
                tuplas_grafo=resposta_grafo,
            )
            st.markdown("### Resposta do grafo:")
            st.write(resposta_grafo)
            st.markdown("### Resposta do LLM:")
            st.write(resposta_llm)
        else:
            st.warning("Digite uma pergunta primeiro.")

# TABS --------------------------------------------------------------------------
st.markdown("---")

modo_extracao = st.session_state.get("metodo_atual", "Simbólico")

tab1, tab2, tab3 = st.tabs([
    "🕸️ Grafo",
    "📌 Arestas",
    "📌 Nós"
])

# GRAFO -------------------------------------------------------------------------
with tab1:
    grafo = st.session_state.grafo
    st.caption(f"Modo atual: {modo_extracao}")
    st.write(f"Nós: **{grafo.number_of_nodes()}** | Arestas: **{grafo.number_of_edges()}**")

    if grafo.number_of_nodes() > 0:
        st.pyplot(grafo_module.desenhar_grafo(grafo))
    else:
        st.warning("O grafo está vazio.")

# TRIPLAS -----------------------------------------------------------------------
with tab2:
    triplas = st.session_state.triplas
    st.caption(f"Modo atual: {modo_extracao}")

    if triplas:
        triplas_arestas = [
            {
                "Evento (Ação)": t["origem"],
                "Relação Semântica": t["papel"],
                "Objeto": t["destino"]
            } for t in triplas if t["tipo"] == "aresta"
        ]   
        st.dataframe(
            pd.DataFrame(triplas_arestas, columns=["Evento (Ação)", "Relação Semântica", "Objeto"]),
            width='stretch'
        )

        st.markdown("### Formato textual")

        for tripla in triplas:
            if tripla['tipo'] == 'aresta':
                s = tripla['origem']
                p = tripla['papel']
                o = tripla['destino']
                st.code(f"[{s}] --({p})--> [{o}]")
    else:
        st.warning("Nenhuma tripla extraída.")

# NÓS ---------------------------------------------------------------------------
with tab3:
    triplas = st.session_state.triplas
    st.caption(f"Modo atual: {modo_extracao}")

    if triplas:

        triplas_nos = [
            {
                "ID do evento": t["id"],
                "Tipo do evento": t["attrs"]["tipo_evento"]
            } for t in triplas if t["tipo"] == "no"
        ]
        st.dataframe(
            pd.DataFrame(triplas_nos, columns=["ID do evento", "Tipo do evento"]),
            width='stretch'
        )

        st.markdown("### Formato textual")

        for tripla in triplas:
            if tripla['tipo'] == 'no':
                s = tripla['id']
                t = tripla['attrs']['tipo_evento']
                st.code(f"[{s}] (tipo: {t})")
    else:
        st.warning("Nenhuma tripla extraída.")
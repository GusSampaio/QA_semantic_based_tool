import re
import streamlit as st
import spacy
import pandas as pd

from src.auxiliares import limpar_texto, separar_frases, normalizar_termo
from src.extracoes import extrair_triplas_frames
import src.grafo as grafo_module

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
            "Execute no terminal: python -m spacy download pt_core_news_sm"
        )
        st.stop()

nlp = carregar_modelo_spacy()

# Texto de exemplo
TEXTO_PADRAO = """As células-filhas foram geradas pela mitose. A mitose é um processo celular."""
    
func_extracao = extrair_triplas_frames
modo_extracao = "Frames Semânticos"

# Inicialização do estado -------------------------------------------------------
if "triplas" not in st.session_state:
    frases_iniciais = separar_frases(limpar_texto(TEXTO_PADRAO), nlp)
    triplas_iniciais = extrair_triplas_frames(frases_iniciais, nlp)
    grafo_inicial = grafo_module.construir_grafo(triplas_iniciais, nlp)

    st.session_state.frases = frases_iniciais
    st.session_state.triplas = triplas_iniciais
    st.session_state.grafo = grafo_inicial

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
        triplas = func_extracao(frases, nlp)
        grafo = grafo_module.construir_grafo(triplas, nlp)

        st.session_state.frases = frases
        st.session_state.triplas = triplas
        st.session_state.grafo = grafo

        st.success(f"Texto processado com modo: {modo_extracao}")

# COLUNA PERGUNTAS --------------------------------------------------------------
with col2:
    st.header("💬 Pergunte ao grafo")
    pergunta = st.text_input("Digite uma pergunta:", placeholder="O que é mitose?")

    if st.button("Responder"):
        if pergunta.strip():
            st.info(grafo_module.responder_pergunta(pergunta, st.session_state.grafo))
        else:
            st.warning("Digite uma pergunta primeiro.")

    st.markdown("### Consulta direta")
    conceito_consulta = st.text_input("Conceito:", placeholder="mitose")
    acao_consulta = st.text_input("Ação (opcional):", placeholder="instancia_de")

    if st.button("Consultar"):
        if conceito_consulta.strip():
            acao = normalizar_termo(acao_consulta) if acao_consulta.strip() else None
            respostas = grafo_module.fazer_pergunta_ao_grafo(
                st.session_state.grafo,
                conceito_consulta,
                acao
            )
            if respostas:
                for obj, ac in respostas:
                    st.write(f"**{normalizar_termo(conceito_consulta)}** --({ac})--> **{obj}**")
            else:
                st.warning("Nenhuma relação encontrada.")
        else:
            st.warning("Digite um conceito.")

# TABS --------------------------------------------------------------------------
st.markdown("---")

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
                "Sujeito": t["origem"],
                "Predicado": t["papel"],
                "Objeto": t["destino"]
            } for t in triplas if t["tipo"] == "aresta"
        ]   
        st.dataframe(
            pd.DataFrame(triplas_arestas, columns=["Sujeito", "Predicado", "Objeto", "Origem"]),
            width='stretch'
        )

        st.markdown("### Formato textual")
        for tripla in triplas:
            print(tripla)

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
            print(tripla)

        for tripla in triplas:
            if tripla['tipo'] == 'no':
                s = tripla['id']
                t = tripla['attrs']['tipo_evento']
                st.code(f"[{s}] (tipo: {t})")
    else:
        st.warning("Nenhuma tripla extraída.")
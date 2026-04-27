import re
import streamlit as st
import spacy
import pandas as pd

from src.auxiliares import limpar_texto, separar_frases, normalizar_termo
from src.extracoes import extrair_triplas
import src.regras as regras_module
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

# Texto de exemplo para processamento inicial
TEXTO_PADRAO = """As células-filhas foram geradas pela mitose. A mitose é um processo celular."""

triplas = extrair_triplas(separar_frases(limpar_texto(TEXTO_PADRAO), nlp), nlp)
grafo = grafo_module.construir_grafo(triplas, nlp)

# Inicializacao do estado da aplicacao
if "triplas" not in st.session_state:
    frases_iniciais = separar_frases(limpar_texto(TEXTO_PADRAO), nlp)
    triplas_iniciais = extrair_triplas(frases_iniciais, nlp)
    grafo_inicial = grafo_module.construir_grafo(triplas_iniciais, nlp)

    st.session_state.frases = frases_iniciais
    st.session_state.triplas = triplas_iniciais
    st.session_state.grafo = grafo_inicial

# Definicoes de interface e logica de processamento
st.title("📖 Livro Didático Virtual Interativo")

st.markdown("""
Pipeline: **texto → segmentação → árvore UD → regra copulativa → grafo**
""")

with st.sidebar:
    st.header("💬 Perguntas de exemplo")
    st.write("• O que é mitose?")
    st.write("• O que é prófase?")
    st.write("• O que é fuso mitótico?")
    st.markdown("---")
    st.caption("Apenas a regra copulativa está ativa nesta versão.")


col1, col2 = st.columns([1, 1])

with col1:
    st.header("📖 Capítulo")
    texto = st.text_area("Cole ou edite o capítulo:", value=TEXTO_PADRAO, height=370)

    if st.button("🔎 Processar texto e gerar grafo"):
        frases = separar_frases(limpar_texto(texto), nlp)
        triplas = extrair_triplas(frases, nlp)
        grafo = grafo_module.construir_grafo(triplas, nlp)

        st.session_state.frases = frases
        st.session_state.triplas = triplas
        st.session_state.grafo = grafo
        st.success("Texto processado!")

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
            respostas = grafo_module.fazer_pergunta_ao_grafo(st.session_state.grafo, conceito_consulta, acao)
            if respostas:
                for obj, ac in respostas:
                    st.write(f"**{normalizar_termo(conceito_consulta)}** --({ac})--> **{obj}**")
            else:
                st.warning("Nenhuma relação encontrada.")
        else:
            st.warning("Digite um conceito.")


st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "🕸️ Grafo",
    "📌 Triplas",
    "🧩 Análise por frase",
    "🧠 Como funciona"
])

with tab1:
    grafo = st.session_state.grafo
    st.write(f"Nós: **{grafo.number_of_nodes()}** | Arestas: **{grafo.number_of_edges()}**")
    if grafo.number_of_nodes() > 0:
        st.pyplot(grafo_module.desenhar_grafo(grafo))
    else:
        st.warning("O grafo está vazio.")

with tab2:
    triplas = st.session_state.triplas
    if triplas:
        st.dataframe(
            pd.DataFrame(triplas, columns=["Sujeito", "Predicado", "Objeto", "Origem"]),
            width='stretch'
        )
        st.markdown("### Formato textual")
        for s, p, o, orig in triplas:
            st.code(f"[{s}] --({p})--> [{o}]")
    else:
        st.warning("Nenhuma tripla extraída.")

with tab3:
    st.caption("Mostra o que a regra copulativa encontrou em cada frase.")
    for i, frase in enumerate(st.session_state.frases, start=1):
        st.write(f"**{i}.** {frase}")
        doc = nlp(frase)
        triplas_frase = regras_module.extrair_copula(frase, doc)
        if triplas_frase:
            for s, p, o, _ in triplas_frase:
                st.code(f"sujeito:    {s}\npredicado:  {p}\nobjeto:     {o}")
        else:
            st.caption("→ Nenhuma construção copulativa encontrada.")

with tab4:
    st.markdown("""
    ### A regra copulativa

    Frases copulativas ("X é Y") têm uma estrutura específica no UD
    onde o verbo "ser" **não é o ROOT** — ele é marcado como `cop` (cópula).
    O ROOT é o predicativo (o que vem depois do "é").

    **Exemplo: "A mitose é um processo celular."**
    ```
    processo  ← ROOT
    ├── mitose   dep: nsubj   → sujeito
    ├── é        dep: cop     → sinal da copulativa
    └── celular  dep: amod
    ```

    **Passos da regra:**
    1. Encontra token com `dep == cop`
    2. O HEAD desse token é o predicativo
    3. Busca irmão com `dep == nsubj` → sujeito
    4. Gera tripla com predicado baseado no POS do predicativo:
       - NOUN → `instancia_de`
       - ADJ  → `tem_propriedade`
    """)
import re
import streamlit as st
import spacy
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# 1. CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Livro Didático Interativo com Grafo Semântico",
    page_icon="🧬",
    layout="wide"
)


# ============================================================
# 2. CARREGAMENTO DO MODELO SPACY
# ============================================================

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


# ============================================================
# 3. TEXTO PADRÃO
# ============================================================

TEXTO_PADRAO = """
A mitose é um processo de divisão celular.
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
# 4. FUNÇÕES DE LIMPEZA E SEGMENTAÇÃO
# ============================================================

def limpar_texto(texto: str) -> str:
    texto = texto.replace("\n", " ")
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


def separar_frases(texto: str):
    doc = nlp(texto)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def normalizar_termo(termo: str) -> str:
    termo = termo.lower().strip()
    termo = re.sub(r"[.!?;:]+$", "", termo)
    termo = re.sub(r"\s+", " ", termo)
    termo = termo.strip()
    termo = re.sub(r"^(a|o|as|os|uma|um|umas|uns)\s+", "", termo)
    return termo


# ============================================================
# 5. EXTRATOR DE PAPÉIS SEMÂNTICOS (baseado em dependências UD)
#
# Regras derivadas empiricamente do Porttinari-base PropBank
# (8.418 sentenças, 39.603 anotações token-papel):
#
#   nsubj       → Arg0  (agente)          77% dos casos
#   nsubj:pass  → Arg1  (paciente)        92% dos casos
#   obl:agent   → Arg0  (ag. de passiva)  95% dos casos
#   obj         → Arg1  (paciente/tema)   92% dos casos
#   ccomp       → Arg1  (comp. oracional) 97% dos casos
#   iobj        → Arg2  (destinatário)    79% dos casos
#   advmod+neg  → ArgM-neg (negação)      99% dos casos
#   obl         → Arg2  (parcial — ambíguo, refinamento futuro)
#
# Referência: Porttinari-base PropBank (Freitas & Pardo, 2024/2025)
# ============================================================

LEXICOS_NEGACAO = {"não", "nem", "jamais", "nunca"}


def extrair_papeis_semanticos(frase: str) -> list[dict]:
    """
    Para cada predicado verbal da frase, retorna um dicionário com os
    papéis semânticos identificados pelas regras simbólicas sobre UD.

    Retorna lista de dicts no formato:
    {
        "predicado":  str,        # lema do verbo
        "Arg0":       str | None, # agente
        "Arg1":       str | None, # paciente / tema
        "Arg2":       str | None, # argumento 2 (destinatário / obl genérico)
        "ArgM-neg":   str | None, # negação
        "ArgM-tmp":   str | None, # (reservado — refinamento futuro via obl)
        "ArgM-loc":   str | None, # (reservado — refinamento futuro via obl)
    }
    """
    doc = nlp(frase)

    # Coleta todos os predicados verbais da frase
    predicados = [t for t in doc if t.pos_ == "VERB"]
    if not predicados:
        # fallback: usa AUX se não houver VERB
        predicados = [t for t in doc if t.pos_ == "AUX"]

    resultados = []

    for pred in predicados:
        estrutura = {
            "predicado": normalizar_termo(pred.lemma_),
            "Arg0":      None,
            "Arg1":      None,
            "Arg2":      None,
            "ArgM-neg":  None,
            "ArgM-tmp":  None,
            "ArgM-loc":  None,
        }

        for token in doc:
            # Só considera dependentes diretos do predicado atual
            if token.head != pred:
                continue

            span = normalizar_termo(" ".join(t.text for t in token.subtree))

            dep = token.dep_

            # ── Arg0: agente ──────────────────────────────────────────
            # nsubj → Arg0 em 77% | obl:agent → Arg0 em 95%
            if dep == "nsubj":
                estrutura["Arg0"] = span

            elif dep == "obl:agent":
                # agente explícito de voz passiva ("foi feito POR ele")
                estrutura["Arg0"] = span

            # ── Arg1: paciente / tema ─────────────────────────────────
            # nsubj:pass → Arg1 em 92% (sujeito em voz passiva é paciente)
            elif dep == "nsubj:pass":
                estrutura["Arg1"] = span

            # obj → Arg1 em 92%
            elif dep == "obj":
                estrutura["Arg1"] = span

            # ccomp / ccomp:speech → Arg1 em 96-99%
            elif dep in ("ccomp", "ccomp:speech"):
                estrutura["Arg1"] = span

            # xcomp → Arg1 em 70% | Arg2 em 29%
            # Heurística: se já temos Arg1, trata como Arg2; senão Arg1
            elif dep == "xcomp":
                if estrutura["Arg1"] is None:
                    estrutura["Arg1"] = span
                elif estrutura["Arg2"] is None:
                    estrutura["Arg2"] = span

            # ── Arg2: destinatário / argumento periférico ─────────────
            # iobj → Arg2 em 79%
            elif dep == "iobj":
                estrutura["Arg2"] = span

            # obl → distribuído entre Arg2/ArgM-tmp/ArgM-loc/ArgM-mnr
            # Por ora mapeado para Arg2 como aproximação conservadora
            # (refinamento futuro: inspecionar preposição subordinada)
            elif dep == "obl":
                if estrutura["Arg2"] is None:
                    estrutura["Arg2"] = span

            # ── ArgM-neg: negação ─────────────────────────────────────
            # advmod + lema de negação → ArgM-neg em 99%
            elif dep == "advmod" and token.lemma_.lower() in LEXICOS_NEGACAO:
                estrutura["ArgM-neg"] = token.text

        resultados.append(estrutura)

    return resultados


# ============================================================
# 6. CONVERSÃO DE ESTRUTURAS SRL → TRIPLAS PARA O GRAFO
#
# Mapeamento de papéis semânticos para arestas do grafo:
#
#   Arg0 --[predicado]--> Arg1   (relação principal agente→paciente)
#   Arg0 --[predicado]--> Arg2   (quando Arg1 ausente)
#   Arg1 --[sofre_acao]--> Arg2  (Arg2 como destinatário de Arg1)
#   ArgM-neg modifica o rótulo do predicado: predicado_neg
# ============================================================

def estruturas_para_triplas(estruturas: list[dict], origem: str = "srl_ud") -> list[tuple]:
    """
    Converte lista de estruturas SRL em triplas (sujeito, predicado, objeto, origem)
    para inserção no grafo de conhecimento.
    """
    triplas = []

    for est in estruturas:
        pred = est["predicado"]
        arg0 = est["Arg0"]
        arg1 = est["Arg1"]
        arg2 = est["Arg2"]
        neg  = est["ArgM-neg"]

        # Se há negação, marca o predicado
        if neg:
            pred = f"{pred}_neg"

        # Relação principal: Arg0 → Arg1
        if arg0 and arg1:
            triplas.append((arg0, pred, arg1, origem))

        # Arg0 → Arg2 quando não há Arg1
        elif arg0 and arg2:
            triplas.append((arg0, pred, arg2, origem))

        # Sem Arg0: Arg1 → Arg2 (ex: passivas sem agente explícito)
        elif arg1 and arg2:
            triplas.append((arg1, f"tem_{pred}", arg2, origem))

    return triplas


# ============================================================
# 7. EXTRAÇÃO FINAL DE TRIPLAS
# ============================================================

def extrair_triplas(frases, usar_srl_ud=True) -> list[tuple]:
    """
    Produz triplas no formato (sujeito, predicado, objeto, origem)
    a partir da lista de frases, usando as regras SRL sobre UD.
    """
    triplas = []

    for frase in frases:
        frase_limpa = frase.strip()

        if usar_srl_ud:
            estruturas = extrair_papeis_semanticos(frase_limpa)
            triplas.extend(estruturas_para_triplas(estruturas))

    # Remove duplicatas por (sujeito, predicado, objeto)
    vistas = set()
    triplas_unicas = []
    for sujeito, predicado, objeto, origem in triplas:
        chave = (sujeito, predicado, objeto)
        if chave not in vistas:
            vistas.add(chave)
            triplas_unicas.append((sujeito, predicado, objeto, origem))

    return triplas_unicas


# ============================================================
# 8. CLASSIFICAÇÃO DE NÓS (baseada em POS via spaCy)
# ============================================================

def classificar_no(no: str) -> str:
    """
    Classifica o tipo semântico de um nó usando o POS do token principal.
    Mantém lógica simples e genérica — não assume domínio.
    """
    doc = nlp(no)
    pos_set = {t.pos_ for t in doc if t.pos_ not in ("DET", "PUNCT", "SPACE")}

    if "PROPN" in pos_set:
        return "Entidade nomeada"
    if "VERB" in pos_set:
        return "Evento / Processo"
    if "ADJ" in pos_set and "NOUN" not in pos_set:
        return "Propriedade"
    if "NOUN" in pos_set:
        return "Conceito"
    return "Outro"


# ============================================================
# 9. CONSTRUÇÃO DO GRAFO
# ============================================================

def construir_grafo(triplas: list[tuple]) -> nx.DiGraph:
    grafo = nx.DiGraph()

    for sujeito, predicado, objeto, origem in triplas:
        grafo.add_node(sujeito, tipo=classificar_no(sujeito))
        grafo.add_node(objeto,  tipo=classificar_no(objeto))
        grafo.add_edge(sujeito, objeto, acao=predicado, origem=origem)

    return grafo


# ============================================================
# 10. PERGUNTAS AO GRAFO
# ============================================================

def fazer_pergunta_ao_grafo(grafo, conceito, acao_desejada=None):
    respostas = []
    conceito = normalizar_termo(conceito)

    if conceito in grafo:
        for vizinho in grafo.successors(conceito):
            dados = grafo.get_edge_data(conceito, vizinho)
            if acao_desejada is None or dados["acao"] == acao_desejada:
                respostas.append((vizinho, dados["acao"]))

    return respostas


def detectar_conceito_na_pergunta(pergunta, grafo):
    pergunta_norm = normalizar_termo(pergunta)
    nos = list(grafo.nodes)
    for no in sorted(nos, key=len, reverse=True):
        if no in pergunta_norm:
            return no
    return None


def responder_pergunta(pergunta, grafo):
    pergunta_norm = normalizar_termo(pergunta)
    conceito = detectar_conceito_na_pergunta(pergunta_norm, grafo)

    if not conceito:
        return (
            "Não encontrei no grafo um conceito mencionado nessa pergunta. "
            "Tente usar termos presentes no texto processado."
        )

    # O que é X? → busca predicado "ser" saindo de X
    if pergunta_norm.startswith("o que é") or "defina" in pergunta_norm or "explique" in pergunta_norm:
        respostas = fazer_pergunta_ao_grafo(grafo, conceito, "ser")
        if respostas:
            return f"{conceito.capitalize()} é {', '.join(r[0] for r in respostas)}."

    # Onde ocorre X? → busca ArgM-loc (por ora via obl no Arg2)
    if "onde" in pergunta_norm or "ocorre" in pergunta_norm:
        respostas = fazer_pergunta_ao_grafo(grafo, conceito, "ocorrer")
        if respostas:
            return f"{conceito.capitalize()} ocorre em: {', '.join(r[0] for r in respostas)}."

    # Para que serve X? / Qual a função de X?
    if "serve" in pergunta_norm or "função" in pergunta_norm or "funcao" in pergunta_norm:
        respostas = fazer_pergunta_ao_grafo(grafo, conceito, "ajudar")
        if respostas:
            return f"{conceito.capitalize()} serve para: {', '.join(r[0] for r in respostas)}."

    # Resposta genérica: todas as relações saindo do conceito
    respostas = fazer_pergunta_ao_grafo(grafo, conceito, None)
    if respostas:
        partes = [f"{acao} → {obj}" for obj, acao in respostas]
        return f"Relações encontradas para '{conceito}': " + "; ".join(partes) + "."

    return "Encontrei o conceito no grafo, mas não há relações suficientes para responder."


# ============================================================
# 11. VISUALIZAÇÃO DO GRAFO
# ============================================================

def desenhar_grafo(grafo):
    fig, ax = plt.subplots(figsize=(14, 9))
    pos = nx.spring_layout(grafo, seed=42, k=1.2)

    # Cor dos nós por tipo
    CORES_TIPO = {
        "Entidade nomeada": "#4e79a7",
        "Evento / Processo": "#f28e2b",
        "Propriedade":       "#59a14f",
        "Conceito":          "#76b7b2",
        "Outro":             "#bab0ac",
    }
    cores = [CORES_TIPO.get(grafo.nodes[n].get("tipo", "Outro"), "#bab0ac") for n in grafo.nodes]

    nx.draw_networkx_nodes(grafo, pos, node_color=cores, node_size=2200, ax=ax)
    nx.draw_networkx_edges(grafo, pos, arrows=True, arrowstyle="->", arrowsize=18, ax=ax)
    nx.draw_networkx_labels(grafo, pos, font_size=8, ax=ax)
    nx.draw_networkx_edge_labels(
        grafo, pos,
        edge_labels=nx.get_edge_attributes(grafo, "acao"),
        font_size=7, ax=ax
    )
    ax.axis("off")

    # Legenda
    from matplotlib.patches import Patch
    legend = [Patch(color=c, label=t) for t, c in CORES_TIPO.items()]
    ax.legend(handles=legend, loc="lower left", fontsize=7)

    return fig


# ============================================================
# 12. INICIALIZAÇÃO DO ESTADO
# ============================================================

if "triplas" not in st.session_state:
    frases_iniciais = separar_frases(limpar_texto(TEXTO_PADRAO))
    triplas_iniciais = extrair_triplas(frases_iniciais)
    grafo_inicial = construir_grafo(triplas_iniciais)

    st.session_state.frases = frases_iniciais
    st.session_state.triplas = triplas_iniciais
    st.session_state.grafo = grafo_inicial


# ============================================================
# 13. INTERFACE STREAMLIT
# ============================================================

st.title("🧬 Livro Didático Virtual Interativo")
st.subheader("SRL simbólico sobre dependências UD + grafo de conhecimento")

st.markdown(
    """
Este protótipo extrai papéis semânticos no estilo **PropBank** (Arg0, Arg1, Arg2, ArgM-neg...)
usando regras simbólicas sobre a árvore de dependências UD produzida pelo spaCy.

As regras foram derivadas empiricamente do corpus **Porttinari-base PropBank**
(Freitas & Pardo, 2024/2025 — 8.418 sentenças em português).

Pipeline: **texto → segmentação → dependências UD → papéis semânticos → triplas → grafo**
"""
)

with st.sidebar:
    st.header("⚙️ Configurações")

    usar_srl_ud = st.checkbox(
        "Usar extração SRL sobre dependências UD",
        value=True
    )

    st.markdown("---")
    st.header("💬 Perguntas de exemplo")
    st.write("• O que é mitose?")
    st.write("• Onde ocorre a mitose?")
    st.write("• Para que serve o fuso mitótico?")
    st.markdown("---")
    st.caption(
        "Regras baseadas em: Porttinari-base PropBank (Freitas & Pardo, 2024/2025). "
        "obl ainda é mapeado como Arg2 genérico — refinamento via preposição em desenvolvimento."
    )


col1, col2 = st.columns([1, 1])

with col1:
    st.header("📖 Capítulo")

    texto = st.text_area(
        "Cole ou edite o capítulo:",
        value=TEXTO_PADRAO,
        height=370
    )

    if st.button("🔎 Processar texto e gerar grafo"):
        texto_limpo = limpar_texto(texto)
        frases = separar_frases(texto_limpo)
        triplas = extrair_triplas(frases, usar_srl_ud=usar_srl_ud)
        grafo = construir_grafo(triplas)

        st.session_state.frases = frases
        st.session_state.triplas = triplas
        st.session_state.grafo = grafo
        st.success("Texto processado e grafo gerado com sucesso!")


with col2:
    st.header("💬 Pergunte ao grafo")

    pergunta = st.text_input(
        "Digite uma pergunta:",
        placeholder="Exemplo: O que é mitose?"
    )

    if st.button("Responder"):
        if pergunta.strip():
            resposta = responder_pergunta(pergunta, st.session_state.grafo)
            st.markdown("### Resposta")
            st.info(resposta)
        else:
            st.warning("Digite uma pergunta primeiro.")

    st.markdown("### Consulta direta ao grafo")

    conceito_consulta = st.text_input(
        "Conceito de origem:",
        placeholder="Exemplo: mitose"
    )
    acao_consulta = st.text_input(
        "Ação desejada (opcional):",
        placeholder="Exemplo: ser, ocorrer, ajudar"
    )

    if st.button("Consultar grafo diretamente"):
        if conceito_consulta.strip():
            acao = normalizar_termo(acao_consulta) if acao_consulta.strip() else None
            respostas = fazer_pergunta_ao_grafo(st.session_state.grafo, conceito_consulta, acao)

            if respostas:
                st.success("Resultados encontrados:")
                for objeto, acao_encontrada in respostas:
                    st.write(f"**{normalizar_termo(conceito_consulta)}** --({acao_encontrada})--> **{objeto}**")
            else:
                st.warning("Nenhuma relação encontrada.")
        else:
            st.warning("Digite um conceito de origem.")


st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🕸️ Grafo",
    "📌 Triplas",
    "🧩 Papéis semânticos por frase",
    "📊 Nós do grafo",
    "🧠 Como funciona"
])

with tab1:
    st.header("Grafo de conhecimento")
    grafo = st.session_state.grafo
    st.write(f"Total de nós: **{grafo.number_of_nodes()}**")
    st.write(f"Total de arestas: **{grafo.number_of_edges()}**")
    if grafo.number_of_nodes() > 0:
        fig = desenhar_grafo(grafo)
        st.pyplot(fig)
    else:
        st.warning("O grafo está vazio.")

with tab2:
    st.header("Triplas extraídas")
    triplas = st.session_state.triplas
    if triplas:
        df_triplas = pd.DataFrame(
            triplas,
            columns=["Sujeito / Arg0", "Predicado", "Objeto / Arg1+", "Origem"]
        )
        st.dataframe(df_triplas, width='stretch')
        st.markdown("### Formato textual")
        for sujeito, predicado, objeto, origem in triplas:
            st.code(f"[{sujeito}] --({predicado})--> [{objeto}]    origem: {origem}")
    else:
        st.warning("Nenhuma tripla foi extraída.")

with tab3:
    st.header("Papéis semânticos por frase")
    st.caption(
        "Mostra as estruturas SRL intermediárias antes da conversão em triplas do grafo. "
        "Cada predicado verbal gera uma estrutura separada."
    )

    for i, frase in enumerate(st.session_state.frases, start=1):
        st.write(f"**{i}.** {frase}")
        estruturas = extrair_papeis_semanticos(frase)

        if not estruturas:
            st.warning("Nenhum predicado encontrado.")
            continue

        for est in estruturas:
            # Monta exibição apenas com papéis não-nulos
            linhas = [f"Predicado: {est['predicado']}"]
            for papel in ("Arg0", "Arg1", "Arg2", "ArgM-neg", "ArgM-tmp", "ArgM-loc"):
                if est[papel] is not None:
                    linhas.append(f"{papel}: {est[papel]}")
            st.code("\n".join(linhas))

with tab4:
    st.header("Nós do grafo")
    grafo = st.session_state.grafo
    dados_nos = [
        {"Nó": no, "Tipo": dados.get("tipo", "Conceito")}
        for no, dados in grafo.nodes(data=True)
    ]
    if dados_nos:
        st.dataframe(pd.DataFrame(dados_nos), width='stretch')
    else:
        st.warning("Nenhum nó encontrado.")

with tab5:
    st.header("Como funciona")
    st.markdown("""
    ### Pipeline de extração

    1. **Segmentação**: o texto é dividido em frases pelo spaCy.
    2. **Parsing UD**: o spaCy produz uma árvore de dependências Universal Dependencies para cada frase.
    3. **Regras SRL simbólicas**: para cada predicado verbal, regras sobre as relações UD identificam os papéis semânticos:
        - `nsubj` → **Arg0** (agente) — 77% de precisão no Porttinari-base
        - `nsubj:pass` → **Arg1** (paciente em passiva) — 92%
        - `obl:agent` → **Arg0** (agente de passiva explícito) — 95%
        - `obj` → **Arg1** (paciente/tema) — 92%
        - `ccomp` → **Arg1** (complemento oracional) — 97%
        - `iobj` → **Arg2** (destinatário) — 79%
        - `advmod` + negação → **ArgM-neg** — 99%
        - `obl` → **Arg2** genérico (⚠️ ambíguo — refinamento em desenvolvimento)
    4. **Triplas**: as estruturas SRL são convertidas em triplas `(Arg0, predicado, Arg1)` para o grafo.
    5. **Grafo**: as triplas formam um grafo dirigido consultável.

    ### Limitação atual
    A relação `obl` é a mais ambígua: pode ser ArgM-tmp (tempo), ArgM-loc (lugar),
    ArgM-mnr (modo) ou Arg2. O refinamento via análise da preposição subordinada está em desenvolvimento.
    """)
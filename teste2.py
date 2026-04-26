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

    # Remove artigos iniciais simples
    termo = re.sub(r"^(a|o|as|os|uma|um|umas|uns)\s+", "", termo)

    return termo


def limpar_paciente(texto: str) -> str:
    texto = normalizar_termo(texto)
    return texto


# ============================================================
# 5. EXTRATOR DE PAPÉIS SEMÂNTICOS COM SPACY
# ============================================================

def extrair_papeis_semanticos(frase: str):
    """
    Extrai uma tripla no estilo:
    agente, predicado, paciente

    A lógica segue o exemplo fornecido:
    - agente: sujeito da frase
    - predicado: verbo principal no lema
    - paciente: objeto ou complemento relevante
    """

    doc = nlp(frase)

    agente = None
    predicado = None
    paciente = None

    # 1. Identificar sujeito/agente
    for token in doc:
        if "subj" in token.dep_:
            agente = " ".join([t.text for t in token.subtree])
            agente = normalizar_termo(agente)
            break

    # 2. Identificar verbo principal/predicado
    for token in doc:
        if token.pos_ in ["VERB", "AUX"]:
            # Preferimos verbo lexical quando possível
            if token.pos_ == "VERB":
                predicado = token.lemma_
                break

    # Caso não tenha encontrado VERB, usa AUX
    if predicado is None:
        for token in doc:
            if token.pos_ == "AUX":
                predicado = token.lemma_
                break

    # 3. Identificar objeto direto ou objeto preposicionado
    for token in doc:
        if "obj" in token.dep_:
            paciente = " ".join([t.text for t in token.subtree])
            paciente = limpar_paciente(paciente)
            break

    # 4. Se não encontrou objeto, tenta atributo/complemento
    # Exemplo: "A mitose é um processo de divisão celular."
    if paciente is None:
        for token in doc:
            if token.dep_ in ["attr", "acomp", "xcomp", "ROOT"] and token.text.lower() not in ["é", "são", "ser"]:
                if token != token.head or token.pos_ in ["NOUN", "ADJ", "PROPN"]:
                    paciente = " ".join([t.text for t in token.subtree])
                    paciente = limpar_paciente(paciente)
                    break

    # 5. Se ainda não encontrou, pega complemento preposicional
    # Exemplo: "A mitose ocorre em células eucarióticas."
    if paciente is None:
        for token in doc:
            if token.dep_ in ["obl", "nmod"]:
                paciente = " ".join([t.text for t in token.subtree])
                paciente = limpar_paciente(paciente)
                break

    if predicado:
        predicado = normalizar_termo(predicado)

    return agente, predicado, paciente


# ============================================================
# 6. REGRAS COMPLEMENTARES PARA BIOLOGIA
# ============================================================

def extrair_regras_biologia(frase: str):
    """
    Regras simbólicas complementares para capturar relações importantes
    que o parser sintático pode não representar bem.
    """

    triplas = []
    f = normalizar_termo(frase)

    # Regra: X é um/uma Y
    # Exemplo: A mitose é um processo de divisão celular.
    padrao = r"^(.*?)\s+é\s+um(?:a)?\s+(.+)$"
    m = re.match(padrao, f)
    if m:
        sujeito = normalizar_termo(m.group(1))
        objeto = normalizar_termo(m.group(2))
        triplas.append((sujeito, "ser", objeto, "regra_X_é_Y"))

    # Regra: X gera Y
    # Exemplo: A mitose gera duas células-filhas.
    padrao = r"^(.*?)\s+gera\s+(.+)$"
    m = re.match(padrao, f)
    if m:
        sujeito = normalizar_termo(m.group(1))
        objeto = normalizar_termo(m.group(2))
        triplas.append((sujeito, "gerar", objeto, "regra_X_gera_Y"))

    # Regra: X ocorre em Y
    # Exemplo: A mitose ocorre em células eucarióticas.
    padrao = r"^(.*?)\s+ocorre\s+em\s+(.+)$"
    m = re.match(padrao, f)
    if m:
        sujeito = normalizar_termo(m.group(1))
        objeto = normalizar_termo(m.group(2))
        triplas.append((sujeito, "ocorrer_em", objeto, "regra_X_ocorre_em_Y"))

    # Regra: X é importante para A, B e C
    # Exemplo: A mitose é importante para crescimento, regeneração e substituição celular.
    padrao = r"^(.*?)\s+é\s+importante\s+para\s+(.+)$"
    m = re.match(padrao, f)
    if m:
        sujeito = normalizar_termo(m.group(1))
        itens = re.split(r",|\se\s", m.group(2))

        for item in itens:
            item = normalizar_termo(item)
            if item:
                triplas.append((sujeito, "ser_importante_para", item, "regra_importante_para"))

    # Regra: X é composta por A, B, C e D
    # Exemplo: A mitose é composta por prófase, metáfase, anáfase e telófase.
    padrao = r"^(.*?)\s+é\s+composta\s+por\s+(.+)$"
    m = re.match(padrao, f)
    if m:
        sujeito = normalizar_termo(m.group(1))
        fases = re.split(r",|\se\s", m.group(2))
        fases = [normalizar_termo(fase) for fase in fases if normalizar_termo(fase)]

        for fase in fases:
            triplas.append((sujeito, "ter_fase", fase, "regra_tem_fase"))

        for i in range(len(fases) - 1):
            triplas.append((fases[i], "preceder", fases[i + 1], "regra_ordem_fases"))
            triplas.append((fases[i + 1], "seguir", fases[i], "regra_ordem_fases"))

    # Regra: Na fase X, Y
    # Exemplo: Na metáfase, os cromossomos se alinham no centro da célula.
    padrao = r"^na\s+(.+?),\s+(.+)$"
    m = re.match(padrao, f)
    if m:
        fase = normalizar_termo(m.group(1))
        evento = normalizar_termo(m.group(2))
        triplas.append((fase, "ter_evento", evento, "regra_evento_na_fase"))

    # Regra: X ajuda a Y
    # Exemplo: O fuso mitótico ajuda a separar as cromátides-irmãs.
    padrao = r"^(.*?)\s+ajuda\s+a\s+(.+)$"
    m = re.match(padrao, f)
    if m:
        sujeito = normalizar_termo(m.group(1))
        objeto = normalizar_termo(m.group(2))
        triplas.append((sujeito, "ajudar_a", objeto, "regra_ajuda_a"))

    # Regra: X carrega/carregam Y
    # Exemplo: Os cromossomos carregam o material genético da célula.
    padrao = r"^(.*?)\s+carregam?\s+(.+)$"
    m = re.match(padrao, f)
    if m:
        sujeito = normalizar_termo(m.group(1))
        objeto = normalizar_termo(m.group(2))
        triplas.append((sujeito, "carregar", objeto, "regra_carregar"))

    return triplas


# ============================================================
# 7. EXTRAÇÃO FINAL DE TRIPLAS
# ============================================================

def extrair_triplas(frases, usar_regras_biologia=True, usar_srl_spacy=True):
    """
    Produz triplas seguindo a lógica:
    agente -- predicado --> paciente

    Cada tripla tem:
    sujeito, predicado, objeto, origem
    """

    triplas = []

    for frase in frases:
        frase_limpa = frase.strip()

        # 1. Extração por spaCy, seguindo a lógica do código enviado
        if usar_srl_spacy:
            agente, predicado, paciente = extrair_papeis_semanticos(frase_limpa)

            if agente and predicado and paciente:
                triplas.append((
                    agente,
                    predicado,
                    paciente,
                    "spacy_sujeito_verbo_objeto"
                ))

        # 2. Extração por regras complementares de biologia
        if usar_regras_biologia:
            triplas_regras = extrair_regras_biologia(frase_limpa)
            triplas.extend(triplas_regras)

    # Remove duplicatas considerando sujeito, predicado e objeto
    vistas = set()
    triplas_unicas = []

    for sujeito, predicado, objeto, origem in triplas:
        chave = (sujeito, predicado, objeto)

        if chave not in vistas:
            vistas.add(chave)
            triplas_unicas.append((sujeito, predicado, objeto, origem))

    return triplas_unicas


# ============================================================
# 8. CLASSIFICAÇÃO SIMPLES DE NÓS
# ============================================================

def classificar_no(no: str) -> str:
    fases = {"prófase", "metáfase", "anáfase", "telófase", "prometáfase"}
    estruturas = {
        "cromossomo",
        "cromossomos",
        "cromátides-irmãs",
        "cromatides-irmas",
        "fuso mitótico",
        "material genético",
        "núcleo",
        "núcleos"
    }

    if no in fases:
        return "Fase da mitose"

    if no == "mitose":
        return "Processo biológico"

    for termo in estruturas:
        if termo in no:
            return "Estrutura biológica"

    if "célula" in no or "células" in no:
        return "Entidade celular"

    if len(no.split()) > 4:
        return "Evento ou descrição"

    return "Conceito"


# ============================================================
# 9. CONSTRUÇÃO DO GRAFO
# ============================================================

def construir_grafo(triplas):
    """
    Constrói o grafo seguindo a lógica do código enviado:

    grafo.add_node(agente, tipo=...)
    grafo.add_node(paciente, tipo=...)
    grafo.add_edge(agente, paciente, acao=predicado)
    """

    grafo = nx.DiGraph()

    for sujeito, predicado, objeto, origem in triplas:
        tipo_sujeito = classificar_no(sujeito)
        tipo_objeto = classificar_no(objeto)

        grafo.add_node(sujeito, tipo=tipo_sujeito)
        grafo.add_node(objeto, tipo=tipo_objeto)

        grafo.add_edge(
            sujeito,
            objeto,
            acao=predicado,
            origem=origem
        )

    return grafo


# ============================================================
# 10. PERGUNTAS AO GRAFO
# ============================================================

def fazer_pergunta_ao_grafo(grafo, conceito, acao_desejada=None):
    """
    Navega no grafo partindo de um conceito e filtrando pela ação.
    É uma adaptação direta da função do código fornecido.
    """

    respostas = []

    conceito = normalizar_termo(conceito)

    if conceito in grafo:
        for vizinho in grafo.successors(conceito):
            dados_da_aresta = grafo.get_edge_data(conceito, vizinho)

            if acao_desejada is None:
                respostas.append((vizinho, dados_da_aresta["acao"]))
            else:
                if dados_da_aresta["acao"] == acao_desejada:
                    respostas.append((vizinho, dados_da_aresta["acao"]))

    return respostas


def detectar_conceito_na_pergunta(pergunta, grafo):
    pergunta_norm = normalizar_termo(pergunta)

    nos = list(grafo.nodes)

    # Tenta encontrar o maior conceito presente na pergunta
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
            "Tente usar termos como mitose, prófase, metáfase, anáfase, telófase, cromossomos ou fuso mitótico."
        )

    # Pergunta: O que é X?
    if pergunta_norm.startswith("o que é") or pergunta_norm.startswith("defina") or "explique" in pergunta_norm:
        respostas = fazer_pergunta_ao_grafo(grafo, conceito, "ser")

        if respostas:
            objetos = [r[0] for r in respostas]
            return f"{conceito.capitalize()} é {', '.join(objetos)}."

    # Pergunta: Quais são as fases de X?
    if "fase" in pergunta_norm or "fases" in pergunta_norm:
        respostas = fazer_pergunta_ao_grafo(grafo, conceito, "ter_fase")

        if respostas:
            fases = [r[0] for r in respostas]
            return f"As fases de {conceito} são: {', '.join(fases)}."

    # Pergunta: O que acontece em X?
    if "acontece" in pergunta_norm or "ocorre" in pergunta_norm:
        respostas = fazer_pergunta_ao_grafo(grafo, conceito, "ter_evento")

        if respostas:
            eventos = [r[0] for r in respostas]
            return f"Em {conceito}, ocorre: {', '.join(eventos)}."

        respostas = fazer_pergunta_ao_grafo(grafo, conceito, "ocorrer_em")

        if respostas:
            locais = [r[0] for r in respostas]
            return f"{conceito.capitalize()} ocorre em: {', '.join(locais)}."

    # Pergunta: Qual fase vem depois de X?
    if "depois" in pergunta_norm or "próxima" in pergunta_norm or "proxima" in pergunta_norm:
        respostas = fazer_pergunta_ao_grafo(grafo, conceito, "preceder")

        if respostas:
            proximas = [r[0] for r in respostas]
            return f"Depois de {conceito}, vem: {', '.join(proximas)}."

    # Pergunta: Qual fase vem antes de X?
    if "antes" in pergunta_norm or "anterior" in pergunta_norm:
        respostas = fazer_pergunta_ao_grafo(grafo, conceito, "seguir")

        if respostas:
            anteriores = [r[0] for r in respostas]
            return f"Antes de {conceito}, vem: {', '.join(anteriores)}."

    # Pergunta: Para que serve X?
    if "serve" in pergunta_norm or "função" in pergunta_norm or "funcao" in pergunta_norm:
        respostas = fazer_pergunta_ao_grafo(grafo, conceito, "ajudar_a")

        if respostas:
            funcoes = [r[0] for r in respostas]
            return f"A função de {conceito} é ajudar a: {', '.join(funcoes)}."

        respostas = fazer_pergunta_ao_grafo(grafo, conceito, "ser_importante_para")

        if respostas:
            finalidades = [r[0] for r in respostas]
            return f"{conceito.capitalize()} é importante para: {', '.join(finalidades)}."

    # Pergunta: Onde ocorre X?
    if "onde" in pergunta_norm:
        respostas = fazer_pergunta_ao_grafo(grafo, conceito, "ocorrer_em")

        if respostas:
            locais = [r[0] for r in respostas]
            return f"{conceito.capitalize()} ocorre em: {', '.join(locais)}."

    # Resposta genérica: mostra todas as relações saindo do conceito
    respostas = fazer_pergunta_ao_grafo(grafo, conceito, None)

    if respostas:
        partes = []
        for objeto, acao in respostas:
            partes.append(f"{acao} → {objeto}")

        return f"Encontrei estas relações para {conceito}: " + "; ".join(partes) + "."

    return "Encontrei o conceito no grafo, mas não encontrei relações suficientes para responder."


# ============================================================
# 11. VISUALIZAÇÃO DO GRAFO
# ============================================================

def desenhar_grafo(grafo):
    fig, ax = plt.subplots(figsize=(14, 9))

    pos = nx.spring_layout(grafo, seed=42, k=1.2)

    nx.draw_networkx_nodes(
        grafo,
        pos,
        node_size=2200,
        ax=ax
    )

    nx.draw_networkx_edges(
        grafo,
        pos,
        arrows=True,
        arrowstyle="->",
        arrowsize=18,
        ax=ax
    )

    nx.draw_networkx_labels(
        grafo,
        pos,
        font_size=8,
        ax=ax
    )

    edge_labels = nx.get_edge_attributes(grafo, "acao")

    nx.draw_networkx_edge_labels(
        grafo,
        pos,
        edge_labels=edge_labels,
        font_size=7,
        ax=ax
    )

    ax.axis("off")
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
st.subheader("PLN simbólico + spaCy + grafo de conhecimento")

st.markdown(
    """
Este protótipo analisa um capítulo de biologia, extrai triplas no formato:

**Agente / Sujeito → Predicado / Ação → Paciente / Objeto**

Depois, transforma essas triplas em um grafo consultável pelo aluno.
"""
)

with st.sidebar:
    st.header("⚙️ Configurações")

    usar_srl_spacy = st.checkbox(
        "Usar extração spaCy sujeito-verbo-objeto",
        value=True
    )

    usar_regras_biologia = st.checkbox(
        "Usar regras complementares de biologia",
        value=True
    )

    st.markdown("---")

    st.header("💬 Perguntas de exemplo")
    st.write("• O que é mitose?")
    st.write("• Quais são as fases da mitose?")
    st.write("• O que acontece na metáfase?")
    st.write("• Qual fase vem depois da metáfase?")
    st.write("• Qual fase vem antes da anáfase?")
    st.write("• Para que serve o fuso mitótico?")
    st.write("• Onde ocorre mitose?")

    st.markdown("---")

    st.caption("Baseado em extração simbólica e dependência sintática.")


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

        triplas = extrair_triplas(
            frases,
            usar_regras_biologia=usar_regras_biologia,
            usar_srl_spacy=usar_srl_spacy
        )

        grafo = construir_grafo(triplas)

        st.session_state.frases = frases
        st.session_state.triplas = triplas
        st.session_state.grafo = grafo

        st.success("Texto processado e grafo gerado com sucesso!")


with col2:
    st.header("💬 Pergunte ao grafo")

    pergunta = st.text_input(
        "Digite uma pergunta:",
        placeholder="Exemplo: O que acontece na metáfase?"
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
        "Ação desejada, opcional:",
        placeholder="Exemplo: ter_fase, ser, preceder"
    )

    if st.button("Consultar grafo diretamente"):
        if conceito_consulta.strip():
            acao = normalizar_termo(acao_consulta) if acao_consulta.strip() else None

            respostas = fazer_pergunta_ao_grafo(
                st.session_state.grafo,
                conceito_consulta,
                acao
            )

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
    "🧩 Frases analisadas",
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
            columns=["Sujeito / Agente", "Predicado / Ação", "Objeto / Paciente", "Origem"]
        )

        st.dataframe(df_triplas, width='stretch')

        st.markdown("### Formato textual")

        for sujeito, predicado, objeto, origem in triplas:
            st.code(f"[{sujeito}] --({predicado})--> [{objeto}]    origem: {origem}")
    else:
        st.warning("Nenhuma tripla foi extraída.")


with tab3:
    st.header("Frases analisadas")

    for i, frase in enumerate(st.session_state.frases, start=1):
        st.write(f"**{i}.** {frase}")

        agente, predicado, paciente = extrair_papeis_semanticos(frase)

        st.code(
            f"Agente: {agente}\n"
            f"Predicado: {predicado}\n"
            f"Paciente: {paciente}"
        )


with tab4:
    st.header("Nós do grafo")

    grafo = st.session_state.grafo

    dados_nos = []

    for no, dados in grafo.nodes(data=True):
        dados_nos.append({
            "Nó": no,
            "Tipo": dados.get("tipo", "Conceito")
        })

    if dados_nos:
        df_nos = pd.DataFrame(dados_nos)
        st.dataframe(df_nos, width='stretch')
    else:
        st.warning("Nenhum nó encontrado.")
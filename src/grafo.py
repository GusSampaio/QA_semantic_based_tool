import networkx as nx
import spacy
import matplotlib.pyplot as plt
from src.auxiliares import normalizar_termo


def classificar_no(no: str, nlp: spacy.Language) -> str:
    doc = nlp(no)
    pos_set = {t.pos_ for t in doc if t.pos_ not in ("DET", "PUNCT", "SPACE")}

    if "PROPN" in pos_set:
        return "Entidade nomeada"
    if "ADJ" in pos_set and "NOUN" not in pos_set:
        return "Propriedade"
    if "NOUN" in pos_set:
        return "Conceito"
    return "Outro"


def construir_grafo(elementos: list, nlp: spacy.Language) -> nx.DiGraph:
    """
    Constrói grafo a partir de elementos estruturados (nós + arestas).
    
    Args:
        elementos: lista de dicts com 'tipo' = 'no' ou 'aresta'
    """
    grafo = nx.DiGraph()

    # Primeiro pass: adiciona todos os nós
    for elem in elementos:
        if elem["tipo"] == "no":
            grafo.add_node(elem["id"], **elem["attrs"])

    # Segundo pass: adiciona arestas
    for elem in elementos:
        if elem["tipo"] == "aresta":
            origem = elem["origem"]
            destino = elem["destino"]
            papel = elem["papel"]

            # Garante que nós dos argumentos existem
            if origem not in grafo:
                grafo.add_node(origem, tipo=classificar_no(origem, nlp))
            if destino not in grafo:
                grafo.add_node(destino, tipo=classificar_no(destino, nlp))

            grafo.add_edge(origem, destino, papel=papel)

    return grafo

def fazer_pergunta_ao_grafo(grafo, conceito, acao_desejada=None):
    respostas = []
    conceito = normalizar_termo(conceito)

    if conceito in grafo:
        for vizinho in grafo.successors(conceito):
            dados = grafo.get_edge_data(conceito, vizinho)
            if acao_desejada is None or dados["papel"] == acao_desejada:
                respostas.append((vizinho, dados["papel"]))

    return respostas


def detectar_conceito_na_pergunta(pergunta, grafo):
    pergunta_norm = normalizar_termo(pergunta)
    for no in sorted(grafo.nodes, key=len, reverse=True):
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

    # O que é X? → busca instancia_de
    if "o que é" in pergunta_norm or "defina" in pergunta_norm:
        respostas = fazer_pergunta_ao_grafo(grafo, conceito, "instancia_de")
        if respostas:
            return f"{conceito.capitalize()} é {', '.join(r[0] for r in respostas)}."

    # Como é X? / Qual a característica de X? → busca tem_propriedade
    if "como é" in pergunta_norm or "característica" in pergunta_norm or "propriedade" in pergunta_norm:
        respostas = fazer_pergunta_ao_grafo(grafo, conceito, "tem_propriedade")
        if respostas:
            return f"{conceito.capitalize()} é {', '.join(r[0] for r in respostas)}."

    # Resposta genérica: todas as relações saindo do conceito
    respostas = fazer_pergunta_ao_grafo(grafo, conceito, None)
    if respostas:
        partes = [f"{acao} → {obj}" for obj, acao in respostas]
        return f"Relações encontradas para '{conceito}': " + "; ".join(partes) + "."

    return "Encontrei o conceito no grafo, mas não há relações suficientes para responder."


# ============================================================
# 10. VISUALIZAÇÃO DO GRAFO
# ============================================================

def desenhar_grafo(grafo):
    fig, ax = plt.subplots(figsize=(14, 9))
    pos = nx.spring_layout(grafo, seed=42, k=1.5)

    CORES_TIPO = {
        "Entidade nomeada": "#4e79a7",
        "Propriedade":      "#59a14f",
        "Conceito":         "#76b7b2",
        "Outro":            "#bab0ac",
    }
    cores = [CORES_TIPO.get(grafo.nodes[n].get("tipo", "Outro"), "#bab0ac") for n in grafo.nodes]

    nx.draw_networkx_nodes(grafo, pos, node_color=cores, node_size=2200, ax=ax)
    nx.draw_networkx_edges(grafo, pos, arrows=True, arrowstyle="->", arrowsize=18, ax=ax)
    nx.draw_networkx_labels(grafo, pos, font_size=8, ax=ax)
    nx.draw_networkx_edge_labels(
        grafo, pos,
        edge_labels=nx.get_edge_attributes(grafo, "papel"),
        font_size=7, ax=ax
    )
    ax.axis("off")

    from matplotlib.patches import Patch
    legend = [Patch(color=c, label=t) for t, c in CORES_TIPO.items()]
    ax.legend(handles=legend, loc="lower left", fontsize=7)

    return fig
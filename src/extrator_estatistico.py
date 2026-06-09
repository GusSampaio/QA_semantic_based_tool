"""
Extração estatística de frames semânticos.

Abordagem: candidatos a argumento são pontuados por dois sinais estatísticos
combinados — frequência relativa do termo no corpus e distância na árvore de
dependências — em vez de regras binárias sobre labels UD específicos.

Diferenças fundamentais em relação ao método simbólico
------------------------------------------------------
Simbólico (frames.py):
  • Detecção binária: um label UD ou bate com a regra ou não.
  • Sem pontuação de qualidade; todos os frames extraídos são equivalentes.
  • Não tenta extrair frame se o sinal esperado (nsubj, obj, cop…) estiver ausente.

Estatístico (este módulo):
  • Cada candidato a argumento recebe um score ∈ (0, 1].
  • Fallback posicional: se não há sinal direto de dependência, usa chunks
    nominais por proximidade linear ao verbo.
  • Limiar de confiança (LIMIAR_CONFIANCA): frames com score insuficiente
    são descartados antes de chegarem ao grafo.
"""

from collections import Counter

from spacy.tokens import Doc, Token

from src.auxiliares import normalizar_termo, eh_tempo
from src.extrator_base import ExtratorBase
import src.frames as frames_module


# ─── Constantes ──────────────────────────────────────────────────────────────

_PREPS_LOCATIVAS = frozenset({"em", "no", "na", "nos", "nas", "sobre", "sob", "entre"})
_PREPS_TEMPORAIS = frozenset({"durante", "após", "antes", "depois", "desde", "até"})

# Desconto aplicado ao score de argumentos herdados de verbos coordenados ou
# identificados pelo fallback posicional (sinais menos confiáveis).
_DESCONTO_CONJ = 0.85
_DESCONTO_FALLBACK = 0.80

# Score mínimo para um frame ser incluído no resultado.
LIMIAR_CONFIANCA: float = 0.20


# ─── Componentes estatísticos ─────────────────────────────────────────────────

def _calcular_frequencias(frases: list, nlp) -> dict:
    """Frequência relativa de cada lema nominal no corpus.

    Retorna um dict {lema: score} onde score ∈ (0, 1].
    O lema mais frequente recebe 1.0; os demais são normalizados por ele.
    Termos mais centrais ao texto recebem score maior e, portanto, são
    preferidos quando competem como candidatos a argumento.

    Exemplo (texto sobre mitose):
        "mitose"       → aparece 5×  → score ≈ 1.00
        "célula"       → aparece 3×  → score ≈ 0.60
        "eucariontes"  → aparece 1×  → score ≈ 0.20
    """
    contagem: dict = Counter()
    for frase in frases:
        doc = nlp(frase)
        for token in doc:
            if token.pos_ in ("NOUN", "PROPN") and not token.is_stop:
                contagem[token.lemma_.lower()] += 1

    if not contagem:
        return {}

    max_count = max(contagem.values())
    return {lema: cnt / max_count for lema, cnt in contagem.items()}


def _distancia_arvore(tok_a: Token, tok_b: Token) -> int:
    """Número de arestas no caminho mais curto entre dois tokens na árvore UD.

    Algoritmo: sobe da raiz de cada token registrando o índice de cada nó
    visitado; encontra o ancestral comum mais baixo (LCA) e soma os dois
    trechos do caminho.
    """
    def caminho_ate_raiz(tok: Token) -> list:
        caminho = []
        visitados: set = set()
        atual = tok
        while atual.i not in visitados:
            visitados.add(atual.i)
            caminho.append(atual.i)
            if atual.head.i == atual.i:  # raiz aponta para si mesma
                break
            atual = atual.head
        return caminho

    caminho_a = caminho_ate_raiz(tok_a)
    caminho_b = caminho_ate_raiz(tok_b)
    indice_b = {no: i for i, no in enumerate(caminho_b)}

    for i, no in enumerate(caminho_a):
        if no in indice_b:
            return i + indice_b[no]

    return len(tok_a.doc)  # fallback para tokens em subárvores desconexas


def _score_candidato(token: Token, verbo: Token, freq: dict) -> float:
    """Pontuação estatística de um token como candidato a argumento de 'verbo'.

    Combina dois sinais de naturezas distintas:
    - freq_score  : frequência do lema no corpus  (sinal léxico-distribucional)
    - prox_score  : 1 / (1 + distância na árvore) (sinal sintático-estrutural)

    O valor 0.05 é o score padrão para termos ausentes no corpus, evitando
    que argumentos de frases curtas sejam descartados prematuramente.
    """
    freq_score = min(freq.get(token.lemma_.lower(), 0.05), 1.0)
    prox_score = 1.0 / (1.0 + _distancia_arvore(token, verbo))
    return 0.5 * freq_score + 0.5 * prox_score


def _resolver_relativo(token: Token) -> Token:
    """Retorna o antecedente nominal de um pronome relativo ('que', 'qual'…).

    Sobe na árvore UD até encontrar um NOUN/PROPN. Se não encontrar,
    retorna o próprio token inalterado.
    """
    if token.text.lower() not in ("que", "quem", "qual", "cujo", "cuja"):
        return token

    atual = token.head
    visitados: set = set()
    while atual.i not in visitados:
        visitados.add(atual.i)
        if atual.pos_ in ("NOUN", "PROPN"):
            return atual
        if atual.head.i == atual.i:
            break
        atual = atual.head
    return token


def _extrair_span_chunk(token: Token, doc: Doc) -> str:
    """Extrai e normaliza o chunk nominal que contém o token.

    Usa os chunks nominais do spaCy (calculados pelo modelo estatístico)
    para obter a expressão referencial completa do argumento, excluindo
    determinantes. Resolve pronomes relativos antes da busca no chunk.
    """
    token = _resolver_relativo(token)

    for chunk in doc.noun_chunks:
        if chunk.start <= token.i < chunk.end:
            partes = [
                t.text for t in chunk
                if t.pos_ != "DET" and t.dep_ != "det"
            ]
            if partes:
                return normalizar_termo(" ".join(partes))

    return normalizar_termo(token.text)


def _extrair_modificadores(verbo: Token, doc: Doc) -> dict:
    """Extrai modificadores locativos e temporais do verbo.

    Aplica a mesma classificação de preposições usada no método simbólico,
    mas via `_extrair_span_chunk` em vez de subtree completa.
    """
    argms: dict = {"loc": [], "tmp": [], "outros": []}
    for filho in verbo.children:
        if filho.dep_ not in ("obl", "advmod"):
            continue
        span_text = _extrair_span_chunk(filho, doc)
        preps = [c.lemma_.lower() for c in filho.children if c.dep_ == "case"]
        if any(p in _PREPS_TEMPORAIS for p in preps) or eh_tempo(span_text):
            argms["tmp"].append(span_text)
        elif any(p in _PREPS_LOCATIVAS for p in preps):
            argms["loc"].append(span_text)
        else:
            argms["outros"].append(span_text)
    return argms


# ─── Extração de frames ───────────────────────────────────────────────────────

def _extrair_frames_verbais(doc: Doc, freq: dict) -> list:
    """Frames verbais em voz ativa e passiva com pontuação estatística.

    Fluxo por verbo:
    1. Coleta candidatos a Arg0 e Arg1 a partir de dependências diretas.
       Cada candidato recebe um score = f(frequência, proximidade).
    2. Para verbos coordenados (conj), herda candidatos do verbo-pai
       com desconto de _DESCONTO_CONJ.
    3. Fallback: se nenhum candidato direto foi encontrado, usa chunks
       nominais à esquerda/direita do verbo com desconto _DESCONTO_FALLBACK.
    4. Seleciona o candidato de maior score para cada papel (Arg0, Arg1).
    5. Descarta o frame se nenhum argumento atingiu LIMIAR_CONFIANCA.
    """
    frames = []
    chunks = list(doc.noun_chunks)

    for token in doc:
        if token.pos_ != "VERB":
            continue

        arg0_cands: list = []  # (score, texto)
        arg1_cands: list = []

        # ── Sinal primário: dependências diretas pontuadas ────────────────
        for filho in token.children:
            texto = _extrair_span_chunk(filho, doc)
            score = _score_candidato(filho, token, freq)

            if filho.dep_ == "nsubj":
                arg0_cands.append((score, texto))
            elif filho.dep_ == "obl:agent":
                arg0_cands.append((score * 0.9, texto))
            elif filho.dep_ == "obj":
                arg1_cands.append((score, texto))
            elif filho.dep_ == "nsubj:pass":
                arg1_cands.append((score, texto))

        # ── Herança de argumentos em verbos coordenados ───────────────────
        if token.dep_ == "conj":
            for irmao in token.head.children:
                texto = _extrair_span_chunk(irmao, doc)
                score = _score_candidato(irmao, token, freq) * _DESCONTO_CONJ
                if irmao.dep_ == "nsubj" and not arg0_cands:
                    arg0_cands.append((score, texto))
                elif irmao.dep_ == "obj" and not arg1_cands:
                    arg1_cands.append((score, texto))

        # ── Fallback estatístico: chunks nominais por posição ─────────────
        # Ativado apenas quando nenhum sinal de dependência foi encontrado.
        # Premissa distribucional: o sujeito (Arg0) tende a aparecer antes
        # do verbo e o objeto (Arg1) depois — válido para SVO em português.
        if not arg0_cands and not arg1_cands:
            for chunk in chunks:
                score = _score_candidato(chunk.root, token, freq) * _DESCONTO_FALLBACK
                if chunk.end <= token.i:
                    arg0_cands.append((score, normalizar_termo(chunk.text)))
                elif chunk.start > token.i:
                    arg1_cands.append((score, normalizar_termo(chunk.text)))

        # ── Seleção do melhor candidato acima do limiar ───────────────────
        frame: dict = {
            "verbo": token,
            "predicado": normalizar_termo(token.lemma_),
            "Arg0": None,
            "Arg1": None,
            "Arg2": None,
            "ArgMs": _extrair_modificadores(token, doc),
        }

        if arg0_cands:
            melhor_s, melhor_t = max(arg0_cands, key=lambda x: x[0])
            if melhor_s >= LIMIAR_CONFIANCA:
                frame["Arg0"] = melhor_t

        if arg1_cands:
            melhor_s, melhor_t = max(arg1_cands, key=lambda x: x[0])
            if melhor_s >= LIMIAR_CONFIANCA:
                frame["Arg1"] = melhor_t

        if frame["Arg0"] or frame["Arg1"]:
            frames.append(frame)

    return frames


def _extrair_frames_copula(doc: Doc, freq: dict) -> list:
    """Frames copulativos com pontuação de confiança no sujeito.

    Usa o mesmo padrão de detecção do método simbólico (dep_='cop'), mas
    adiciona avaliação estatística: o sujeito só é aceito se seu score
    atingir LIMIAR_CONFIANCA — filtrando casos onde o parser atribuiu
    uma dependência nsubj fraca ou inesperada.
    """
    frames = []

    for token in doc:
        if token.dep_ != "cop":
            continue

        predicativo = token.head

        # Busca o sujeito com maior score entre os filhos nsubj do predicativo
        sujeito_token = None
        melhor_score = 0.0
        for filho in predicativo.children:
            if filho.dep_ == "nsubj":
                score = _score_candidato(filho, predicativo, freq)
                if score > melhor_score:
                    melhor_score = score
                    sujeito_token = filho

        if sujeito_token is None or melhor_score < LIMIAR_CONFIANCA:
            continue

        sujeito = normalizar_termo(
            " ".join(t.text for t in sujeito_token.subtree)
        )

        # Índices do sujeito a excluir do span do objeto
        indices_sujeito = {t.i for t in sujeito_token.subtree}

        objeto = normalizar_termo(
            " ".join(
                t.text for t in predicativo.subtree
                if t.i not in indices_sujeito
                and t.dep_ != "cop"
                and t.dep_ != "nsubj"
            )
        )

        if not sujeito or not objeto:
            continue

        if predicativo.pos_ in ("NOUN", "PROPN"):
            predicado = "instancia_de"
        elif predicativo.pos_ == "ADJ":
            predicado = "tem_propriedade"
        else:
            predicado = "ser"

        frames.append({
            "verbo": predicativo,
            "predicado": predicado,
            "Arg0": sujeito,
            "Arg1": objeto,
            "Arg2": None,
            "ArgMs": {"loc": [], "tmp": [], "outros": []},
        })

    return frames


def _extrair_todos_frames(doc: Doc, freq: dict) -> list:
    """Combina extração verbal e copulativa (mesma estrutura de frames.py)."""
    frames = _extrair_frames_verbais(doc, freq)
    frames.extend(_extrair_frames_copula(doc, freq))
    return frames


# ─── Classe extratora ─────────────────────────────────────────────────────────

class ExtratorEstatistico(ExtratorBase):
    """Extrator estatístico de frames semânticos.

    Entrada / saída são idênticas ao ExtratorSimbolico; a diferença está
    no processo interno de seleção de argumentos.

    Vantagens sobre o método simbólico
    ------------------------------------
    • Recall potencialmente maior: o fallback posicional permite extrair
      frames mesmo quando o parser falha em atribuir nsubj/obj explícitos.
    • Priorização por relevância: termos centrais ao texto (alta frequência)
      são preferidos como argumentos.
    • Rejeição de ruído: frames com score baixo são descartados antes de
      contaminar o grafo.

    Limitações
    ----------
    • Em textos com uma única frase, a frequência de termos tem pouca
      variação, reduzindo o poder discriminativo do sinal léxico.
    • O fallback posicional pode falhar em orações com ordem SOV ou
      estruturas marcadas (tópico-comentário, foco, etc.).
    • Não realiza inferência lógica sobre os frames extraídos — assim
      como o método simbólico, é puramente extrativista.
    """

    def extrair_elementos(self, frases: list, nlp) -> list:
        # Computa frequências uma única vez sobre todo o corpus
        freq = _calcular_frequencias(frases, nlp)

        elementos = []
        event_id = 0
        for frase in frases:
            doc = nlp(frase.strip())
            frames = _extrair_todos_frames(doc, freq)
            # Reutiliza frames_para_grafo_estruturado de frames.py
            novos_elementos, event_id = frames_module.frames_para_grafo_estruturado(
                frames, event_id
            )
            elementos.extend(novos_elementos)

        return elementos

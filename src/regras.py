import spacy
from src.auxiliares import normalizar_termo

def extrair_copula(doc: spacy.Language) -> list[tuple]:
    triplas = []

    for token in doc:
        # Passo 1: encontra o "é", "são", "era"... (dep == cop)
        if token.dep_ != "cop":
            continue

        # Passo 2: o head desse token é o predicativo
        predicativo = token.head

        # Passo 3: busca o sujeito — irmão do predicativo com dep == nsubj
        sujeito_token = None
        for irmao in predicativo.head.children if predicativo.dep_ == "ROOT" else predicativo.children:
            if irmao.dep_ == "nsubj":
                sujeito_token = irmao
                break

        # Fallback: busca nsubj diretamente nos filhos do predicativo
        if sujeito_token is None:
            for filho in predicativo.children:
                if filho.dep_ == "nsubj":
                    sujeito_token = filho
                    break

        if sujeito_token is None:
            continue

        # Monta os spans normalizados
        sujeito = normalizar_termo(
            " ".join(t.text for t in sujeito_token.subtree)
        )
        objeto = normalizar_termo(
            " ".join(t.text for t in predicativo.subtree
                     # exclui o sujeito e a cópula do span do predicativo
                     if t not in list(sujeito_token.subtree)
                     and t.dep_ != "cop"
                     and t.dep_ != "nsubj")
        )

        if not sujeito or not objeto:
            continue

        # Passo 4: diferencia o predicado pelo POS do predicativo
        if predicativo.pos_ in ("NOUN", "PROPN"):
            predicado = "instancia_de"
        elif predicativo.pos_ == "ADJ":
            predicado = "tem_propriedade"
        else:
            predicado = "ser"

        triplas.append((sujeito, predicado, objeto, "regra_copula"))

    return triplas

def extrair_objeto_direto(doc: spacy.Language) -> list[tuple]:
    """
    Regra do objeto direto (dep == obj).
    
    Cobre dois casos:

    VOZ ATIVA:
      - Arg0: irmão com dep == nsubj
      - predicado: lema do verbo
      - Arg1: token com dep == obj
      Exemplo: "A mitose gera células-filhas."
        → [mitose] --(gerar)--> [células-filhas]

    VOZ PASSIVA:
      - Arg1: irmão com dep == nsubj:pass  (sujeito paciente)
      - Arg0: irmão com dep == obl:agent   (agente explícito, opcional)
      - predicado: lema do verbo
      Exemplo: "As células-filhas foram geradas pela mitose."
        → [mitose] --(gerar)--> [células-filhas]

    Baseada em:
      obj        → Arg1 em 92,03% dos casos
      nsubj:pass → Arg1 em 92%   dos casos
      obl:agent  → Arg0 em 95%   dos casos
    (Porttinari-base PropBank, Freitas & Pardo, 2024/2025)
    """
    triplas = []
    for token in doc:
        if token.dep_ != "obj" and token.dep_ != "nsubj:pass":
            continue

        verbo = token.head
        if verbo.pos_ not in ("VERB", "AUX"):
            continue

        # Coleta filhos relevantes do verbo
        nsubj_token      = None
        nsubj_pass_token = None
        obl_agent_token  = None

        for irmao in verbo.children:
            if irmao.dep_ == "nsubj":
                nsubj_token = irmao
            elif irmao.dep_ == "nsubj:pass":
                nsubj_pass_token = irmao
            elif irmao.dep_ == "obl:agent":
                obl_agent_token = irmao

        predicado = normalizar_termo(verbo.lemma_)
        objeto    = normalizar_termo(" ".join(t.text for t in token.subtree))

        # ── Voz ativa ──────────────────────────────────────────────────
        if nsubj_token is not None:
            sujeito = normalizar_termo(" ".join(t.text for t in nsubj_token.subtree))
            triplas.append((sujeito, predicado, objeto, "regra_obj_ativa"))

        # ── Voz passiva ────────────────────────────────────────────────
        elif nsubj_pass_token is not None:
            paciente = normalizar_termo(" ".join(t.text for t in nsubj_pass_token.subtree))

            if obl_agent_token is not None:
                # Agente explícito: "As células foram geradas pela mitose"
                agente = normalizar_termo(" ".join(t.text for t in obl_agent_token.subtree))
                triplas.append((agente, predicado, paciente, "regra_obj_passiva_com_agente"))
            else:
                # Sem agente explícito: registra só o paciente com Arg0 desconhecido
                triplas.append((None, predicado, paciente, "regra_obj_passiva_sem_agente"))

    return triplas


def extrair_obl(doc: spacy.Language) -> list[tuple]:
    triplas = []

    for verbo in doc:
        if verbo.pos_ != "VERB":
            continue

        nsubj = None
        nsubj_pass = None

        for filho in verbo.children:
            if filho.dep_ == "nsubj":
                nsubj = filho
            elif filho.dep_ == "nsubj:pass":
                nsubj_pass = filho

        predicado = normalizar_termo(verbo.lemma_)

        for filho in verbo.children:
            if filho.dep_ != "obl":
                continue

            span = normalizar_termo(" ".join(t.text for t in filho.subtree))

            # pega preposição
            prep = None
            for t in filho.children:
                if t.dep_ == "case":
                    prep = t.text.lower()
                    break

            if prep in {"em", "no", "na", "nos", "nas"}:
                papel = "ArgM-loc"
            elif prep in {"durante", "após", "antes"} or (prep == "em" and any(t.like_num for t in filho.subtree)):
                papel = "ArgM-tmp"
            elif prep is not None:
                papel = "ArgM"

            if nsubj is not None:
                arg0 = normalizar_termo(" ".join(t.text for t in nsubj.subtree))
            elif nsubj_pass is not None:
                arg0 = None  # ou você pode decidir outra estratégia
            else:
                arg0 = None

            triplas.append((arg0, predicado, span, papel))

    return triplas
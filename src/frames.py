import spacy
from src.auxiliares import normalizar_termo

def criar_frame(verbo):
    return {
        "verbo": verbo,
        "predicado": normalizar_termo(verbo.lemma_),
        "Arg0": None,
        "Arg1": None,
        "Arg2": None,
        "ArgMs": {
            "loc": [],
            "tmp": [],
            "outros": []
        }
    }

def extrair_frames_copula(doc):
    frames = []

    for token in doc:
        if token.dep_ != "cop":
            continue

        predicativo = token.head

        # acha sujeito
        sujeito_token = None
        for filho in predicativo.children:
            if filho.dep_ == "nsubj":
                sujeito_token = filho
                break

        if sujeito_token is None:
            continue

        sujeito = normalizar_termo(" ".join(t.text for t in sujeito_token.subtree))

        objeto = normalizar_termo(
            " ".join(
                t.text for t in predicativo.subtree
                if t not in list(sujeito_token.subtree)
                and t.dep_ != "cop"
                and t.dep_ != "nsubj"
            )
        )

        if not sujeito or not objeto:
            continue

        # define predicado
        if predicativo.pos_ in ("NOUN", "PROPN"):
            predicado = "instancia_de"
        elif predicativo.pos_ == "ADJ":
            predicado = "tem_propriedade"
        else:
            predicado = "ser"

        frame = {
            "verbo": predicativo,  # só pra manter padrão
            "predicado": predicado,
            "Arg0": sujeito,
            "Arg1": objeto,
            "Arg2": None,
            "ArgMs": {"loc": [], "tmp": [], "outros": []}
        }

        frames.append(frame)

    return frames

def preencher_objeto_direto(frame):
    verbo = frame["verbo"]

    nsubj = None
    nsubj_pass = None
    obl_agent = None

    for filho in verbo.children:
        if filho.dep_ == "nsubj":
            nsubj = filho
        elif filho.dep_ == "nsubj:pass":
            nsubj_pass = filho
        elif filho.dep_ == "obl:agent":
            obl_agent = filho

    for filho in verbo.children:
        if filho.dep_ == "obj":
            frame["Arg1"] = normalizar_termo(" ".join(t.text for t in filho.subtree))

        elif filho.dep_ == "nsubj:pass":
            frame["Arg1"] = normalizar_termo(" ".join(t.text for t in filho.subtree))

    # Arg0
    if nsubj is not None:
        frame["Arg0"] = normalizar_termo(" ".join(t.text for t in nsubj.subtree))
    elif obl_agent is not None:
        frame["Arg0"] = normalizar_termo(" ".join(t.text for t in obl_agent.subtree if t.dep_ != "case"))

def preencher_obl(frame):
    verbo = frame["verbo"]

    for filho in verbo.children:
        if filho.dep_ != "obl":
            continue

        span = normalizar_termo(" ".join(t.text for t in filho.subtree))

        prep = None
        for t in filho.children:
            if t.dep_ == "case":
                prep = t.text.lower()
                break

        if prep in {"em", "no", "na", "nos", "nas"}:
            frame["ArgMs"]["loc"].append(span)
        elif prep in {"em", "durante", "após", "antes"}:
            frame["ArgMs"]["tmp"].append(span)
        else:
            frame["ArgMs"]["outros"].append(span)

def extrair_frames(doc):
    frames = []

    for token in doc:
        if token.pos_ != "VERB":
            continue

        frame = criar_frame(token)

        preencher_objeto_direto(frame)
        preencher_obl(frame)

        frames.append(frame)

    return frames

def extrair_todos_frames(doc):
    frames = []

    frames.extend(extrair_frames(doc))        # verbos
    frames.extend(extrair_frames_copula(doc)) # cópula

    return frames

def frames_para_triplas(frames):
    triplas = []

    for f in frames:
        if f["Arg0"] and f["Arg1"]:
            triplas.append((f["Arg0"], f["predicado"], f["Arg1"]))

        for loc in f["ArgMs"]["loc"]:
            triplas.append((f["Arg0"], f["predicado"], loc))

        for tmp in f["ArgMs"]["tmp"]:
            triplas.append((f["predicado"], "tmp", tmp))
        
        for outro in f["ArgMs"]["outros"]:
            triplas.append((f["predicado"], "mod", outro))

    return triplas
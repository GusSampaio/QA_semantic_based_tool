import spacy
from src.auxiliares import normalizar_termo, eh_tempo

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

# Descobre quem são os partipantes do evento e depois preenche o frame (arg0 e arg1)
def preencher_objeto_direto(frame):

    # Centro do frame é o verbo
    verbo = frame["verbo"]

    nsubj = None
    obl_agent = None

    # Descobre quais palavras dependem diretamente desse verbo
    for filho in verbo.children:
        if filho.dep_ == "nsubj": # sujeito ativo (agente)
            nsubj = filho
        elif filho.dep_ == "obl:agent": # agente em voz passiva 
            obl_agent = filho

    # Descobre o objeto direto (paciente)
    for filho in verbo.children:
        if filho.dep_ == "obj":
            frame["Arg1"] = normalizar_termo(" ".join(t.text for t in filho.subtree))
        elif filho.dep_ == "nsubj:pass":
            frame["Arg1"] = normalizar_termo(" ".join(t.text for t in filho.subtree))
    
    if verbo.dep_ == "conj":
        # tenta pegar o objeto direto do verbo coordenado
        for filho in verbo.head.children:
            if filho.dep_ == "obj":
                frame["Arg1"] = normalizar_termo(" ".join(t.text for t in filho.subtree))
                break
            elif filho.dep_ == "nsubj:pass":
                frame["Arg1"] = normalizar_termo(" ".join(t.text for t in filho.subtree))
                break

    # Define o agente (Arg0) - primeiro tenta o sujeito ativo, depois o agente em voz passiva, e por fim tenta pegar o sujeito do verbo coordenado (caso o verbo seja uma conjunção)
    if nsubj is not None:
        frame["Arg0"] = normalizar_termo(" ".join(t.text for t in nsubj.subtree))
    elif obl_agent is not None:
        frame["Arg0"] = normalizar_termo(" ".join(t.text for t in obl_agent.subtree if t.dep_ != "case"))
    elif verbo.dep_ == "conj":
        # tenta pegar o sujeito do verbo coordenado
        for filho in verbo.head.children:
            if filho.dep_ == "nsubj":
                frame["Arg0"] = normalizar_termo(" ".join(t.text for t in filho.subtree))
                break

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
            if eh_tempo(span):
                frame["ArgMs"]["tmp"].append(span)
            else:
                frame["ArgMs"]["loc"].append(span)

        elif prep in {"durante", "após", "antes"}:
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
    event_id = 0

    for f in frames:
        event_node = f"evento_{event_id}"
        event_id += 1

        # tipo do evento (verbo)
        triplas.append((event_node, "tipo", f["predicado"]))

        # Arg0
        if f["Arg0"]:
            triplas.append((event_node, "Arg0", f["Arg0"]))

        # Arg1
        if f["Arg1"]:
            triplas.append((event_node, "Arg1", f["Arg1"]))

        # Arg2 (se quiser já deixar pronto)
        if f["Arg2"]:
            triplas.append((event_node, "Arg2", f["Arg2"]))

        # ArgMs
        for loc in f["ArgMs"]["loc"]:
            triplas.append((event_node, "loc", loc))

        for tmp in f["ArgMs"]["tmp"]:
            triplas.append((event_node, "tmp", tmp))

        for outro in f["ArgMs"]["outros"]:
            triplas.append((event_node, "mod", outro))

    return triplas

def frames_para_grafo_estruturado(frames, event_id_inicial=0):
    """
    Converte frames em estrutura preparada para construção de grafo.
    
    Retorna:
        lista de dicts, cada um representando um elemento do grafo:
        - {"tipo": "no", "id": str, "attrs": dict}  → nó
        - {"tipo": "aresta", "origem": str, "destino": str, "papel": str}  → aresta
    """
    elementos = []
    event_id = event_id_inicial

    for f in frames:
        event_node = f"evento_{event_id}"
        event_id += 1

        # Nó do evento com atributo 'tipo'
        elementos.append({
            "tipo": "no",
            "id": event_node,
            "attrs": {"tipo_evento": f["predicado"]}
        })

        # Arg0 → evento
        if f["Arg0"]:
            elementos.append({
                "tipo": "aresta",
                "origem": f["Arg0"],
                "destino": event_node,
                "papel": "Arg0"
            })

        # evento → Arg1
        if f["Arg1"]:
            elementos.append({
                "tipo": "aresta",
                "origem": event_node,
                "destino": f["Arg1"],
                "papel": "Arg1"
            })

        # evento → Arg2
        if f["Arg2"]:
            elementos.append({
                "tipo": "aresta",
                "origem": event_node,
                "destino": f["Arg2"],
                "papel": "Arg2"
            })

        # evento → ArgM-loc
        for loc in f["ArgMs"]["loc"]:
            elementos.append({
                "tipo": "aresta",
                "origem": event_node,
                "destino": loc,
                "papel": "ArgM-loc"
            })

        # evento → ArgM-tmp
        for tmp in f["ArgMs"]["tmp"]:
            elementos.append({
                "tipo": "aresta",
                "origem": event_node,
                "destino": tmp,
                "papel": "ArgM-tmp"
            })

        # evento → ArgM (outros)
        for outro in f["ArgMs"]["outros"]:
            elementos.append({
                "tipo": "aresta",
                "origem": event_node,
                "destino": outro,
                "papel": "ArgM"
            })

    return elementos, event_id
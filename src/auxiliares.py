
import re
import spacy

def limpar_texto(texto: str) -> str:
    texto = texto.replace("\n", " ")
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()

def separar_frases(texto: str, nlp: spacy.Language) -> list:
    doc = nlp(texto)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def normalizar_termo(termo: str | spacy.tokens.token.Token) -> str:
    termo = str(termo).lower().strip()
    termo = re.sub(r"[.!?;:]+$", "", termo)
    termo = re.sub(r"\s+", " ", termo)
    termo = termo.strip()
    termo = re.sub(r"^(a|o|as|os|uma|um|umas|uns)\s+", "", termo)
    return termo

def eh_tempo(span: str) -> bool:
    span = span.lower()

    # anos (ex: 1999, 2020)
    if re.search(r"\b\d{4}\b", span):
        return True

    # números com unidades de tempo
    if re.search(r"\b\d+\s*(dias?|meses?|anos?|horas?)\b", span):
        return True

    # palavras típicas de tempo
    palavras_tempo = {
        "hoje", "ontem", "amanhã",
        "antes", "depois", "durante",
        "enquanto", "quando",
        "agora", "então",
        "início", "fim"
    }

    if any(p in span for p in palavras_tempo):
        return True

    return False
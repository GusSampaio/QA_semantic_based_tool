
import re
import spacy

def limpar_texto(texto: str) -> str:
    texto = texto.replace("\n", " ")
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()

def separar_frases(texto: str, nlp: spacy.Language) -> list:
    doc = nlp(texto)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def normalizar_termo(termo: str) -> str:
    termo = termo.lower().strip()
    termo = re.sub(r"[.!?;:]+$", "", termo)
    termo = re.sub(r"\s+", " ", termo)
    termo = termo.strip()
    termo = re.sub(r"^(a|o|as|os|uma|um|umas|uns)\s+", "", termo)
    return termo
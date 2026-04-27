import spacy
def carregar_modelo_spacy():
    try:
        return spacy.load("pt_core_news_sm")
    except OSError:
        print("Modelo não encontrado")

nlp = carregar_modelo_spacy()
frase = "As células-filhas foram geradas pela mitose."
doc = nlp(frase)

for token in doc:
    print(f"{token.text:<20} dep={token.dep_:<15} head={token.head.text:<15} pos={token.pos_}")
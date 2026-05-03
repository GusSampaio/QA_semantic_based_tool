from src.nlp_pipeline import SymbolicFact


def test_symbolic_fact_text_contains_triple():
    fact = SymbolicFact(
        event_id="gerar_0",
        predicate="gerar",
        subject="mitose",
        object="células-filhas",
        arg2=None,
        locations=(),
        times=(),
        modifiers=(),
        sentence="A mitose gera células-filhas.",
    )
    text = fact.as_text()
    assert "mitose --[gerar]--> células-filhas" in text
    assert "Frase de origem" in text

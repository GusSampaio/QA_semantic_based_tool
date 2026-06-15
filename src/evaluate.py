import json
import re

import spacy

from sklearn.metrics import precision_recall_fscore_support

from src.frames import extrair_todos_frames


ROLES = {"ARG0", "ARG1", "ARGM-TMP", "ARGM-LOC"}


def limpar_frase(frase):

    frase = frase.replace("<PRED>", "")
    frase = frase.replace("</PRED>", "")
    frase = frase.replace("<\\/PRED>", "")

    frase = frase.replace("[", "")
    frase = frase.replace("]", "")

    frase = re.sub(r"\s+", " ", frase)

    return frase.strip()


def obter_verbo_alvo(tokens):

    for i, tok in enumerate(tokens):

        if tok == "<PRED>" and i + 1 < len(tokens):
            return tokens[i + 1].lower()

    return None


def selecionar_frame(frames, verbo_alvo):

    if verbo_alvo is None:
        return None

    for frame in frames:

        verbo = frame.get("verbo")

        if verbo is None:
            continue

        if verbo.text.lower() == verbo_alvo:
            return frame

    return None


def gerar_predicoes(frame, doc):

    pred = ["O"] * len(doc)

    if frame is None:
        return pred

    arg0 = frame.get("Arg0_head")
    arg1 = frame.get("Arg1_head")

    if arg0 is not None:
        pred[arg0.i] = "ARG0"

    if arg1 is not None:
        pred[arg1.i] = "ARG1"

    argms = frame.get("ArgMs_heads", {})

    for t in argms.get("tmp", []):
        pred[t.i] = "ARGM-TMP"

    for t in argms.get("loc", []):
        pred[t.i] = "ARGM-LOC"

    return pred


def avaliar_jsonl(caminho):

    print("Carregando spaCy...")
    nlp = spacy.load("pt_core_news_sm")
    print("spaCy carregado.")

    y_true = []
    y_pred = []

    total = 0

    with open(caminho, encoding="utf8") as f:

        for linha in f:

            total += 1

            if total % 100 == 0:
                print(f"{total} exemplos processados")

            registro = json.loads(linha)

            frase = limpar_frase(
                registro["frase_original"]
            )

            verbo_alvo = obter_verbo_alvo(
                registro["bert_tokens"]
            )

            doc = nlp(frase)

            frames = extrair_todos_frames(doc)
            if total < 10:
                print("\n====================")
                print("FRASE:")
                print(frase)

                print("\nVERBO ALVO:")
                print(verbo_alvo)

                print("\nFRAMES:")

                for f in frames:
                    print({
                        "verbo": f["verbo"].text,
                        "Arg0": f["Arg0"],
                        "Arg1": f["Arg1"],
                        "ArgMs": f["ArgMs"]
                    })

            frame = selecionar_frame(
                frames,
                verbo_alvo
            )

            predicoes = gerar_predicoes(
                frame,
                doc
            )

            idx_doc = 0

            for gold in registro["labels_verdadeiras"]:

                if gold == "IGNORADO":
                    continue

                if gold == "PRED":
                    idx_doc += 1
                    continue

                if idx_doc >= len(predicoes):
                    break

                if gold in ROLES or gold == "O":

                    y_true.append(gold)
                    y_pred.append(predicoes[idx_doc])

                idx_doc += 1

    p, r, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(ROLES),
        average="micro",
        zero_division=0
    )

    print()
    print("===== GLOBAL =====")
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1:        {f1:.4f}")

    print()
    print("===== POR PAPEL =====")

    for role in sorted(ROLES):

        p_r, r_r, f_r, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[role],
            average=None,
            zero_division=0
        )

        print(
            f"{role:<10} "
            f"P={p_r[0]:.4f} "
            f"R={r_r[0]:.4f} "
            f"F1={f_r[0]:.4f}"
        )


if __name__ == "__main__":
    avaliar_jsonl("dataset_teste.jsonl")
"""Referencia do parser do spaCy: para cada texto grava tokens, cabecas, rotulos, inicios de sentenca e as entidades do
pipeline COMPLETO (com parser). O teste Kotlin (ParserParityTest) exige o mesmo.

Uso:  python scripts/parity/make_parser_fixtures.py
"""
import json
import random
from pathlib import Path

import spacy

ROOT = Path(__file__).resolve().parents[2]
MODELS = {"en": "en_core_web_sm", "pt": "pt_core_news_sm"}

NAMES = {
    "en": ["Pedro", "Maria", "John", "Anna Lee", "Silva", "Peter Parker", "Laura Mendes", "Dr. Brown", "Mr. Jones", "Kauan"],
    "pt": ["Pedro", "Maria", "João", "Ana Lima", "Silva", "Pedro Nascimento", "Laura Mendes", "Dr. Costa", "Sra. Pereira", "Kauan"],
}
TEMPLATES = {
    "en": [
        "Thanks, {a}. {b} said we should wait.", "Hi {a}. {b} is here too.", "I am {a}. {b} is my friend.",
        "{a}. {b}", "Okay {a}, what do you think? {b} thinks it is fine.", "Hello everyone, I'm {a}. And you are {b}?",
        "So {a} and {b} met yesterday. They talked about the project.", "{a} said hello. {b} said goodbye. Then it was over.",
        "Good morning {a}. Good morning {b}. Let us begin.", "Thank you, {a}.  {b} will speak next.\nAnd then we finish.",
    ],
    "pt": [
        "Obrigado, {a}. {b} disse que devemos esperar.", "Oi {a}. {b} também está aqui.", "Eu sou {a}. {b} é meu amigo.",
        "{a}. {b}", "Tudo bem {a}, o que você acha? {b} acha que está bom.", "Bom dia a todos, eu sou {a}. E você é {b}?",
        "Então {a} e {b} se encontraram ontem. Eles conversaram sobre o projeto.", "{a} disse olá. {b} disse tchau. Depois acabou.",
        "Bom dia {a}. Bom dia {b}. Vamos começar.", "Obrigada, {a}.  {b} fala em seguida.\nE depois terminamos.",
    ],
}


def main():
    fx = json.load(open(ROOT / "src/test/resources/parity/ner_fixtures.json", encoding="utf-8"))
    rng = random.Random(5)
    out = {}
    for lang, model in MODELS.items():
        nlp = spacy.load(model)
        base = [c["text"] for c in fx[lang]]
        texts = list(base)
        for _ in range(250):
            texts.append(" ".join(rng.sample(base, rng.randint(2, 4))))
        for t in TEMPLATES[lang]:
            for _ in range(12):
                a, b = rng.sample(NAMES[lang], 2)
                texts.append(t.format(a=a, b=b))
        texts += ["", " ", "a", "Hello.", "  leading and trailing  ", "\n\n", "Wait... what?! No way -- really?", "😀 emoji 🙂 test ¡Hola! ¿Qué tal?"]
        cases = []
        for text in texts:
            doc = nlp(text)
            cases.append({
                "text": text,
                "tokens": [t.text for t in doc],
                "heads": [t.head.i for t in doc],
                "deps": [t.dep_ for t in doc],
                "sents": [bool(t.is_sent_start) for t in doc],
                "ents": [[e.start, e.end, e.label_] for e in doc.ents],
            })
        out[lang] = cases
        print(lang, len(cases), "textos;", sum(1 for c in cases if sum(c["sents"]) > 1), "com mais de uma sentenca")
    (ROOT / "src/test/resources/parity/parser_fixtures.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

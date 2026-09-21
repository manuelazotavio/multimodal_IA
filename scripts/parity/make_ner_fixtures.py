"""Fixtures do spaCy para o NerParityTest (Kotlin): tokens, atributos, entidades e nomes, por idioma."""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import spacy
from spacy.attrs import NORM, PREFIX, SHAPE, SUFFIX
from spacy.strings import hash_string

ROOT = Path(__file__).resolve().parents[2]
AV = Path("C:/Users/manu/av-tracker")
PT = set("que não uma com para isso mas então tá né aí eu você ele ela gente também porque muito já foi tem vou pra está são como mais só bem essa esse".split())

TRICKY = [
    "", " ", "  ", "a", "Pedro", "Hello  world", "tab\there", "line\nbreak", "trailing space ", " leading space",
    "I'm David and this is Mrs. Robinson from Microsoft.", "Don't you think it's e.g. fine? (Yes.) I cannot; won't.",
    "Visit https://example.com/a?b=1 or mail me at pedro.silva@example.com, ok?", "It costs $5.99 or 10% off at 5pm on 3/4/2026.",
    "state-of-the-art well-known Anna-Maria O'Neil U.S.A. Ph.D. a.m. p.m. Dr. Smith", "Wait... what?! No way -- really?",
    "Meu nome é João da Silva, 30 anos, nº 5, 1º andar; d'água pra você.", "Olha, Manuela! O Heitor disse: \"vem cá\" (rápido).",
    "Eu sou o Gustavo Henrique, prazer.", "Fala, Arthur. Sua vez, Kauan. Valeu, Ana!", "Thanks, Pedro. Silva said we should wait.",
    "Hey Laura! Ana, come here. João is late.", "My name is Laura Smith and I work with Pedro Almeida at Google in Paris.",
    "Sr. Carlos e a Dra. Beatriz chegaram; Mr. Jones e Ms. Lee também.", "😀 emoji 🙂 test ¡Hola! ¿Qué tal? naïve café Zoë",
    "x" * 120, "1,000,000.50 and 3.14 and -5 and +7 and 2nd, 3rd; 21st.", "C++ C# .NET node.js @user #hashtag",
    "A 'quoted' word, \"double\" ‘curly’ “curly” «guillemets» (parens) [brackets] {braces}",
]


def is_pt(t):
    return sum(w in PT for w in re.findall(r"\w+", t.lower())) >= 2


lines = []
for f in sorted(glob.glob(str(AV / "realtime_sessions/transcript_*.txt"))):
    for l in open(f, encoding="utf-8", errors="ignore"):
        m = re.match(r"^\[[\d:]+\]\s+[^:]+:\s*(.+)$", l.strip())
        if m and len(m.group(1)) > 5:
            lines.append(m.group(1))

HONORIFIC = re.compile(r"^(Sr\.|Sra\.|Dr\.|Dra\.|Mr\.|Mrs\.|Ms\.)\s+", re.IGNORECASE)
out = {"hashes": {s: str(hash_string(s)) for s in ["", "a", "hello", "Pedro", "João", "xxxx", "Xxxxx", "d", "é", "😀", "abcdefgh", "abcdefghi", "the", "'s", ".", " "]}}
for name, lang in (("en_core_web_sm", "en"), ("pt_core_news_sm", "pt")):
    nlp = spacy.load(name)
    texts = [x for x in lines if is_pt(x) == (lang == "pt")]
    seen, uniq = set(), []
    for x in texts + TRICKY:
        if x not in seen: seen.add(x); uniq.append(x)
    cases = []
    for i, text in enumerate(uniq):
        doc = nlp.make_doc(text)
        arr = doc.to_array([NORM, PREFIX, SUFFIX, SHAPE]).astype(np.uint64) if len(doc) else np.zeros((0, 4), np.uint64)
        ref = nlp.get_pipe("ner")(nlp.make_doc(text))
        names = [HONORIFIC.sub("", e.text.strip()) for e in ref.ents if e.label_ in ("PER", "PERSON") and len(e.text.strip()) > 2]
        case = {"text": text, "tokens": [[t.idx, t.text, t.norm_] for t in doc],
                "ents": [[e.start, e.end, e.label_, e.text] for e in ref.ents], "names": names}
        if i < 80: case["keys"] = [[str(int(v)) for v in row] for row in arr]
        cases.append(case)
    out[lang] = cases
    print(lang, len(cases), "casos")
dest = ROOT / "src/test/resources/parity/ner_fixtures.json"
dest.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
print("gravado", dest, round(dest.stat().st_size / 1024), "KB")

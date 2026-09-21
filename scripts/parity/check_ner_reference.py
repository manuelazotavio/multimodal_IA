"""Valida ner_reference.py (numpy) contra o componente `ner` real do spaCy, no mesmo Doc do tokenizador."""
import glob
import re
import sys
from pathlib import Path

import numpy as np
import spacy
from spacy.attrs import NORM, PREFIX, SHAPE, SUFFIX

sys.path.insert(0, str(Path(__file__).parent))
import ner_reference as R

ROOT = Path(__file__).resolve().parents[2]
AV = Path("C:/Users/manu/av-tracker")
PT = set("que não uma com para isso mas então tá né aí eu você ele ela gente também porque muito já foi tem vou pra está são como mais só bem essa esse".split())


def corpus():
    lines = []
    for f in sorted(glob.glob(str(AV / "realtime_sessions/transcript_*.txt"))):
        for l in open(f, encoding="utf-8", errors="ignore"):
            m = re.match(r"^\[[\d:]+\]\s+[^:]+:\s*(.+)$", l.strip())
            if m and len(m.group(1)) > 5:
                lines.append(m.group(1))
    return lines


def is_pt(t):
    return sum(w in PT for w in re.findall(r"\w+", t.lower())) >= 2


extra = ["My name is Laura Smith and I work with Pedro Almeida at Google in Paris.", "Thanks, Pedro. Silva said we should wait.",
         "Hey Laura! Ana, come here. João is late.", "Olha, Manuela. O Heitor disse que vem. Kauan também.",
         "Fala, Arthur.", "Vamos ouvir a parte do Kauan agora.", "I'm David and this is Mrs. Robinson from Microsoft.",
         "Eu sou o Gustavo Henrique, prazer.", "Dr. Smith met Maria at 5pm on Friday in New York.", "", " ", "a", "Pedro"]

lines = corpus() + extra
total = bad = 0
for name, lang in (("en_core_web_sm", "en"), ("pt_core_news_sm", "pt")):
    nlp = spacy.load(name)
    meta, t = R.load(ROOT / f"src/main/assets/ner/{lang}")
    texts = [x for x in lines if (is_pt(x) == (lang == "pt"))] + extra
    for text in texts:
        doc = nlp.make_doc(text)
        arr = doc.to_array([NORM, PREFIX, SUFFIX, SHAPE]).astype(np.uint64)
        ents = R.run_ner(meta, t, R.tok2vec(meta, t, arr) if len(doc) else np.zeros((0, 64), np.float32), [tok.is_space for tok in doc])
        ref = nlp.get_pipe("ner")(nlp.make_doc(text))
        want = [(e.start, e.end, e.label_) for e in ref.ents]
        total += 1
        if ents != want:
            bad += 1
            if bad <= 5: print("DIFERE", name, repr(text[:80]), "\n   ref:", want, "\n   meu:", ents)
    print(f"{name}: {len(texts)} textos")
print(f"\nTOTAL {total} textos | divergencias: {bad}")

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


_WORD_RE = re.compile(r"[A-Za-z0-9]+")

def tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]


@dataclass
class BM25Index:
    k1: float
    b: float
    avgdl: float
    doc_len: List[int]
    df: Dict[str, int]                 # document frequency
    idf: Dict[str, float]              # idf per term
    tfs: List[Dict[str, int]]          # term freq per document


def build_bm25(
    texts: Sequence[str],
    k1: float = 1.5,
    b: float = 0.75,
) -> BM25Index:
    tfs: List[Dict[str, int]] = []
    df: Dict[str, int] = {}
    doc_len: List[int] = []

    for text in texts:
        toks = tokenize(text)
        doc_len.append(len(toks))
        tf: Dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        tfs.append(tf)

        # update df with unique terms
        for t in tf.keys():
            df[t] = df.get(t, 0) + 1

    N = len(texts)
    avgdl = (sum(doc_len) / N) if N else 0.0

    # BM25+ style IDF (common stable variant)
    idf: Dict[str, float] = {}
    for term, n_qi in df.items():
        # log( (N - df + 0.5) / (df + 0.5) + 1 )
        idf[term] = math.log((N - n_qi + 0.5) / (n_qi + 0.5) + 1.0)

    return BM25Index(k1=k1, b=b, avgdl=avgdl, doc_len=doc_len, df=df, idf=idf, tfs=tfs)


def bm25_search(
    index: BM25Index,
    query: str,
    top_k: int = 50,
) -> Tuple[List[float], List[int]]:
    q_terms = tokenize(query)
    if not q_terms:
        return [], []

    scores: List[float] = [0.0] * len(index.tfs)

    for term in q_terms:
        idf = index.idf.get(term)
        if idf is None:
            continue

        for doc_id, tf in enumerate(index.tfs):
            f = tf.get(term, 0)
            if f == 0:
                continue

            dl = index.doc_len[doc_id]
            denom = f + index.k1 * (1.0 - index.b + index.b * (dl / index.avgdl if index.avgdl else 0.0))
            score = idf * (f * (index.k1 + 1.0)) / (denom if denom else 1.0)
            scores[doc_id] += score

    # top-k by score
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    ranked = [(i, s) for i, s in ranked if s > 0.0][:top_k]

    idx = [i for i, _ in ranked]
    sc = [s for _, s in ranked]
    return sc, idx
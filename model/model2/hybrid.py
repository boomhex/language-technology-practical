from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from bm25 import BM25Index, bm25_search
from similarity import SimilarityIndex, search as dense_search


@dataclass
class HybridConfig:
    k_bm25: int = 100
    k_dense: int = 100
    rrf_k: int = 60          # RRF constant, typical 60
    top_k: int = 10          # final results


def rrf_fuse(
    bm25_ranks: Sequence[int],
    dense_ranks: Sequence[int],
    rrf_k: int = 60,
) -> List[int]:
    """
    Reciprocal Rank Fusion:
      score(d) = Σ 1 / (rrf_k + rank_method(d))
    rank is 1-based.
    """
    scores: Dict[int, float] = {}

    for r, doc_id in enumerate(bm25_ranks, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + r)

    for r, doc_id in enumerate(dense_ranks, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + r)

    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def hybrid_search(
    *,
    embed_model,
    dense: SimilarityIndex,
    bm25: BM25Index,
    query: str,
    cfg: HybridConfig,
) -> List[int]:
    # BM25 candidates (ranked)
    _, bm25_idx = bm25_search(bm25, query=query, top_k=cfg.k_bm25)

    # Dense candidates (ranked)
    _, dense_idx = dense_search(embed_model, dense, query=query, top_k=cfg.k_dense)
    dense_idx_list = [int(i) for i in dense_idx.tolist() if int(i) >= 0]

    fused = rrf_fuse(bm25_idx, dense_idx_list, rrf_k=cfg.rrf_k)
    return fused[: cfg.top_k]
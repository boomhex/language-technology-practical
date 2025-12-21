from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from dataclasses import dataclass
import faiss
import numpy as np

@dataclass
class SimilarityIndex:
    index: faiss.Index
    embeddings: np.ndarray | None
    model_name: str

def _encode_texts(
    model: SentenceTransformer,
    texts: Sequence[str],
    batch_size: int = 64,
    device: str = "cpu",
) -> np.ndarray:
    emb = model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine similarity via inner product
        show_progress_bar=True,
        device=device,
    )
    emb = emb.astype("float32", copy=False)
    return emb

def build_similarity_index(
    texts: Sequence[str],
    enc_model_name: str = "all-MiniLM-L6-v2",
    device: str = "cpu",
    batch_size: int = 64,
) -> Tuple[SentenceTransformer, SimilarityIndex]:
    """
    Builds a FAISS index for cosine similarity using normalized embeddings and IndexFlatIP.
    """
    model = SentenceTransformer(enc_model_name, device=device)
    embeddings = _encode_texts(model, texts, batch_size=batch_size, device=device)

    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)

    return model, SimilarityIndex(index=index, embeddings=embeddings, model_name=enc_model_name)

def search(
    model: SentenceTransformer,
    sim: SimilarityIndex,
    query: str,
    top_k: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (scores, indices) each of shape (top_k,).
    Scores are cosine similarities in [-1, 1] (usually [0,1] with these models).
    """
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    scores, idx = sim.index.search(q, top_k)
    return scores[0], idx[0]
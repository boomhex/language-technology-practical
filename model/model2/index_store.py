from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from load import Chunk
from bm25 import BM25Index

STORE_DIR = Path("schemas")
INDEX_PATH = STORE_DIR / "chunks.faiss"
META_PATH  = STORE_DIR / "chunks_meta.pkl"
def save_store(
    *,
    index: faiss.Index,
    chunks: List[Chunk],
    enc_model_name: str,
    bm25: BM25Index,
) -> None:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))

    payload = {
        "enc_model_name": enc_model_name,
        "chunks": chunks,
        "bm25": bm25,
    }
    with META_PATH.open("wb") as f:
        pickle.dump(payload, f)

def load_store(device: str = "cpu") -> Tuple[SentenceTransformer, faiss.Index, List[Chunk], str, BM25Index]:
    if not INDEX_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError(
            f"Missing saved store. Expected {INDEX_PATH} and {META_PATH}."
        )

    index = faiss.read_index(str(INDEX_PATH))

    with META_PATH.open("rb") as f:
        payload: Dict[str, Any] = pickle.load(f)

    enc_model_name = payload["enc_model_name"]
    chunks = payload["chunks"]
    bm25 = payload["bm25"]

    embed_model = SentenceTransformer(enc_model_name, device=device)
    return embed_model, index, chunks, enc_model_name, bm25
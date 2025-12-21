from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np

from load import load_chunks
from similarity import build_similarity_index, SimilarityIndex
from index_store import save_store, load_store
from bm25 import build_bm25
from hybrid import HybridConfig, hybrid_search

OUT_DIR = Path("schemas")
CHUNKS_PATH = OUT_DIR / "chunks.jsonl"
INDEX_PATH = OUT_DIR / "chunks.faiss"
META_PATH  = OUT_DIR / "chunks_meta.pkl"
REBUILD_STORE = False   # <-- flip to True to force rebuild + save


def main():

    device = "cpu"  # or "cuda:0" if you have a stable GPU setup
    if REBUILD_STORE:
        chunks = load_chunks(CHUNKS_PATH)
        texts = [c.text for c in chunks]

        # Dense index
        embed_model, dense_sim = build_similarity_index(
            texts=texts,
            enc_model_name="all-MiniLM-L6-v2",
            device=device,
            batch_size=16,  # safer on macOS
        )

        # BM25 index
        bm25 = build_bm25(texts)

        # Save both
        save_store(
            index=dense_sim.index,
            chunks=chunks,
            enc_model_name=dense_sim.model_name,
            bm25=bm25,
        )

        dense = dense_sim
    else:
        embed_model, faiss_index, chunks, enc_name, bm25 = load_store(device=device)
        dense = SimilarityIndex(index=faiss_index, embeddings=None, model_name=enc_name)

    cfg = HybridConfig(
        k_bm25=100,  # candidates from BM25
        k_dense=100, # candidates from dense
        rrf_k=60,
        top_k=10,    # final top-k returned
    )

    while True:
        q = input("\nQuery> ").strip()
        if not q:
            continue
        if q.lower() in {"quit", "exit"}:
            break

        top_idxs = hybrid_search(
            embed_model=embed_model,
            dense=dense,
            bm25=bm25,
            query=q,
            cfg=cfg,
        )

        print("\nTop results:")
        for rank, i in enumerate(top_idxs, start=1):
            c = chunks[i]
            md = c.metadata
            text_preview = c.text.replace("\n", " ")[:140]
            print(
                f"{rank:2d}. "
                f"doc_id={md.get('doc_id')} | "
                f"chunk_id={md.get('chunk_id')} | "
                f"section={md.get('section')} | "
                f"dish_name={md.get('dish_name')}"
            )
            print(f"    {text_preview}...\n")


    



if __name__ == "__main__":
    main()
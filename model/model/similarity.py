import numpy as np
from numpy.linalg import norm
import torch
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
from typing import Sequence
# Global models / arrays (for simplicity)


def encode_corpus(model, recipe_chunks, batch_size=32, device="cpu"):
    """
    Encode a list of texts into dense vectors using a SentenceTransformer model.
    Uses a SentenceTransformer model (here, all-MiniLM-L6-v2).
    Here we try to transform each paragraph into a point in high-dimensional space
    """
    all_embeddings = []
    texts = [c["text"] for c in recipe_chunks]
    for start_idx in tqdm(range(0, len(texts), batch_size), desc="Encoding corpus"):
        batch_texts = texts[start_idx:start_idx + batch_size]
        with torch.no_grad():
            emb = model.encode(
                batch_texts,
                convert_to_tensor=True,
                show_progress_bar=False,
                device=device
            )
        all_embeddings.append(emb.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings.numpy().astype("float32")

def build_faiss_index(embeddings):
    """
    Build a simple FAISS index for inner-product similarity.
    For cosine similarity, it's standard to L2-normalize embeddings.
    This helps us build a 'vector database'
    """
    # L2 normalize embeddings
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # IP = inner product
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors of dim {dim}.")
    return index


def build_similarity_index(
    recipe_chunks: Sequence[dict],
    model_name: str = "all-MiniLM-L6-v2",
    device: str = "cpu",
    batch_size: int = 32,
):
 
    embed_model = SentenceTransformer(model_name)
    # Adjust encode_corpus signature if needed; this assumes you can pass batch_size & device
    embeddings = encode_corpus(
        embed_model,
        recipe_chunks,
        batch_size=batch_size,
        device=device,
    )

    index = build_faiss_index(
        embeddings,
    )

    return index, embed_model

from generator import GenerativeQA
import numpy as np
import torch
import faiss
from tqdm import tqdm

def retrieve_top_k(model, index, questions, k=5, batch_size=32, device="cpu"):
    """
    Encode questions and retrieve top-k nearest contexts from the FAISS index.
    Returns:
      - all_scores: [num_questions, k] similarity scores
      - all_indices: [num_questions, k] corpus indices
    We are basically checking which contexts are most semantically similar to the given question
    """
    all_scores = []
    all_indices = []

    for start_idx in tqdm(range(0, len(questions), batch_size), desc="Encoding queries + retrieving"):
        batch_questions = questions[start_idx:start_idx + batch_size]
        with torch.no_grad():
            q_emb = model.encode(
                batch_questions,
                convert_to_tensor=True,
                show_progress_bar=False,
                #device=device
            )
        q_emb = q_emb.cpu().numpy().astype("float32")
        faiss.normalize_L2(q_emb)  # same normalization

        scores, indices = index.search(q_emb, k)
        all_scores.append(scores)
        all_indices.append(indices)

    all_scores = np.vstack(all_scores)
    all_indices = np.vstack(all_indices)
    return all_scores, all_indices

def ask_recipe_question(
    gen_qa: GenerativeQA,
    question: str,
    recipe_chunks,
    embed_model,
    faiss_index,
    k_retrieval: int = 5,
    device: str = "cuda",
):
    """
    Retrieve top-k relevant recipe chunks for a question and generate an answer.

    Returns:
      - answer: str
      - hits: list of dicts with fields {"chunk", "score"}
    """
    
    # 1. retrieve top-k relevant chunks (FAISS + SentenceTransformer)
    scores, indices = retrieve_top_k(
        model=embed_model,
        index=faiss_index,
        questions=[question],        # single-question batch
        k=k_retrieval,
        batch_size=1,
        device=device,
    )

    # scores, indices have shape [1, k]; take the first row
    top_scores = scores[0]
    top_indices = indices[0]

    # build contexts and a structured hits list
    contexts = []
    hits = []
    for rank, (chunk_idx, score) in enumerate(zip(top_indices, top_scores)):
        chunk = recipe_chunks[chunk_idx]
        contexts.append(chunk["text"])
        hits.append(
            {
                "rank": rank,
                "score": float(score),
                "chunk_id": chunk["chunk_id"],
                "chunk": chunk,
            }
        )

    # optional: inspect retrieved chunks
    # for rank, context_text in enumerate(contexts):
    #     print(f"\n--- Retrieved chunk {rank} ---")
    #     print(context_text)

    # 2. generate answer from the retrieved contexts
    answer = gen_qa.answer(question, contexts)
    return answer, hits
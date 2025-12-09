"""
Dense Retrieval with FAISS + SentenceTransformers

What this script does:
- Load SQuAD validation set
- Build a corpus of contexts
- Encode contexts into dense vectors using SentenceTransformers
- Build a FAISS index over these vectors
- Encode questions and retrieve top-k contexts
- Evaluate:
    * Top-k accuracy: is the gold context among the top-k retrieved?
    * Context recall: does the retrieved context contain the gold answer span?
"""

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm



MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
K = 5              # top-k for retrieval
MAX_EXAMPLES = None # subset of examples, change to None if you want to use all


# LOAD DATA


def load_squad_contexts(max_examples=None):
    """
    Load SQuAD validation set and build:
    - contexts: list of unique context strings
    - context_ids: list mapping corpus index -> some ID
    - examples: list of QA examples with question, context_id, answer text
    """
    squad = load_dataset("squad")
    val = squad["validation"]

    if max_examples is not None and max_examples < len(val):
        val = val.select(range(max_examples))

    # Build a mapping from context string to an integer ID
    context_to_id = {}
    contexts = []
    context_ids = []

    examples = []

    for ex in val:
        context = ex["context"]
        if context not in context_to_id:
            context_to_id[context] = len(contexts)
            contexts.append(context)
            context_ids.append(len(contexts) - 1)

        cid = context_to_id[context]

        examples.append({
            "id": ex["id"],
            "question": ex["question"],
            "context": context,
            "context_id": cid,
            "answers": ex["answers"], #gold answer
        })

    print(f"Loaded {len(examples)} QA examples with {len(contexts)} unique contexts.")
    return contexts, examples


# ENCODE CONTEXTS

def encode_corpus(model, texts, batch_size=32, device="cpu"):
    """
    Encode a list of texts into dense vectors using a SentenceTransformer model.
    Uses a SentenceTransformer model (here, all-MiniLM-L6-v2).
    Here we try to transform each paragraph into a point in high-dimensional space
    """
    all_embeddings = []

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


# BUILD FAISS INDEX

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


# ENCODE QUERIES + RETRIEVE

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
                device=device
            )
        q_emb = q_emb.cpu().numpy().astype("float32")
        faiss.normalize_L2(q_emb)  # same normalization

        scores, indices = index.search(q_emb, k)
        all_scores.append(scores)
        all_indices.append(indices)

    all_scores = np.vstack(all_scores)
    all_indices = np.vstack(all_indices)
    return all_scores, all_indices


# EVALUATION METRICS

def compute_top_k_accuracy(examples, retrieved_indices, k, gold_context_ids):
    """
    Top-k accuracy: percentage of questions where the gold context is among the top-k retrieved.
    - examples: list of QA examples
    - retrieved_indices: [num_examples, k] matrix of corpus indices
    - gold_context_ids: list of gold context_id for each example
    """
    assert len(examples) == len(retrieved_indices) == len(gold_context_ids)

    hits = 0
    for i in range(len(examples)):
        gold_cid = gold_context_ids[i]
        if gold_cid in retrieved_indices[i, :k]:
            hits += 1

    return 100.0 * hits / len(examples)


def answer_in_retrieved_contexts(example, retrieved_corpus_indices, corpus_texts):
    """
    Checking whether any of the retrieved contexts contain
    any of the gold answer strings

    Returns True if at least one gold answer text is a substring of at least one
    retrieved context.
    """
    gold_answers = example["answers"]["text"]
    retrieved_texts = [corpus_texts[idx] for idx in retrieved_corpus_indices]

    # Lowercase
    lowered_retrieved = [c.lower() for c in retrieved_texts]

    for ans in gold_answers:
        if ans.strip() == "":
            continue
        ans_lower = ans.lower()
        for ctx in lowered_retrieved:
            if ans_lower in ctx:
                return True

    return False


def compute_context_recall(examples, retrieved_indices, corpus_texts, k):
    """
    Context recall: fraction of questions where the retrieved top-k passages
    contain a gold answer span somewhere in the text.

    This is a simple proxy for "does retrieval bring an answerable context?"
    """
    assert len(examples) == len(retrieved_indices)

    hits = 0
    for i, ex in enumerate(examples):
        topk_indices = retrieved_indices[i, :k]
        if answer_in_retrieved_contexts(ex, topk_indices, corpus_texts):
            hits += 1

    return 100.0 * hits / len(examples)



def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    corpus_texts, examples = load_squad_contexts(max_examples=MAX_EXAMPLES)

    # Gold context IDs for each example
    gold_context_ids = [ex["context_id"] for ex in examples]
    questions = [ex["question"] for ex in examples]

    # Load SentenceTransformer model
    print(f"Loading SentenceTransformer model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device=device)

    # Encode corpus contexts and build FAISS index
    corpus_embeddings = encode_corpus(model, corpus_texts, device=device)
    index = build_faiss_index(corpus_embeddings)

    # Encode questions and retrieve top-k
    scores, indices = retrieve_top_k(model, index, questions, k=K, device=device)

    # Evaluate top-k accuracy
    topk_acc = compute_top_k_accuracy(examples, indices, k=K, gold_context_ids=gold_context_ids)
    print(f"\nTop-{K} retrieval accuracy (gold context in top-{K}): {topk_acc:.2f}%")

    # Evaluate context recall
    ctx_recall = compute_context_recall(examples, indices, corpus_texts, k=K)
    print(f"Context recall (gold answer span in top-{K} contexts): {ctx_recall:.2f}%")


if __name__ == "__main__":
    main()

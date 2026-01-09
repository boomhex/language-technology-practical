import json
from typing import Dict, List, Sequence, Set, Optional, Tuple, Union

import numpy as np
from sentence_transformers import SentenceTransformer


# -----------------------------
# JSONL tags
# -----------------------------
def get_all_tag_categories_from_jsonl(jsonl_path: str) -> List[str]:
    tags: Set[str] = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e
            obj_tags = obj.get("tags")
            if obj_tags:
                tags.update(obj_tags)
    return sorted(tags)


# -----------------------------
# Embeddings / similarity
# -----------------------------
def encode_texts(
    model: SentenceTransformer,
    texts: Sequence[str],
    batch_size: int = 64,
    device: str = "cpu",
    template: Optional[str] = None,
) -> np.ndarray:
    if template is not None:
        texts = [template.format(text=t) for t in texts]

    emb = model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,  # dot == cosine
        show_progress_bar=False,
        device=device,
    )
    return emb.astype(np.float32, copy=False)


def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else (v / n)


WeightedSyn = Union[str, Tuple[str, float]]  # ("synonym", weight)


def build_tag_vectors(
    model: SentenceTransformer,
    tag_to_synonyms: Dict[str, Sequence[WeightedSyn]],
    batch_size: int = 64,
    device: str = "cpu",
    template: str = "This recipe is {text}.",
) -> Dict[str, np.ndarray]:
    """
    Build one stable vector per tag by:
      1) encoding all synonyms (batched)
      2) weighted mean per tag
      3) L2 normalization
    """

    # Flatten all synonyms so we encode once
    flat_texts: List[str] = []
    flat_meta: List[Tuple[str, float]] = []  # (tag, weight)

    for tag, syns in tag_to_synonyms.items():
        cleaned: List[Tuple[str, float]] = []
        for s in syns:
            if isinstance(s, tuple):
                text, w = s
            else:
                text, w = s, 1.0
            text = text.strip()
            if text:
                cleaned.append((text, float(w)))

        if not cleaned:
            raise ValueError(f"Tag '{tag}' has no synonyms.")

        for text, w in cleaned:
            flat_texts.append(text)
            flat_meta.append((tag, w))

    # Encode once
    all_emb = encode_texts(model, flat_texts, batch_size=batch_size, device=device, template=template)

    # Aggregate per tag
    tag_sum: Dict[str, np.ndarray] = {}
    tag_wsum: Dict[str, float] = {}

    for (tag, w), vec in zip(flat_meta, all_emb):
        if tag not in tag_sum:
            tag_sum[tag] = np.zeros_like(vec)
            tag_wsum[tag] = 0.0
        tag_sum[tag] += w * vec
        tag_wsum[tag] += w

    tag_vectors: Dict[str, np.ndarray] = {}
    for tag in tag_sum:
        v = tag_sum[tag] / max(tag_wsum[tag], 1e-12)
        tag_vectors[tag] = l2_normalize(v.astype(np.float32, copy=False))

    return tag_vectors


def score_query_against_tags(
    model: SentenceTransformer,
    query: str,
    tag_vectors: Dict[str, np.ndarray],
    device: str = "cpu",
    template: str = "User request: {text}",
) -> List[Tuple[str, float]]:
    q = encode_texts(model, [query], batch_size=1, device=device, template=template)[0]
    scores = [(tag, float(q @ vec)) for tag, vec in tag_vectors.items()]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def top_k(scores: List[Tuple[str, float]], k: int = 5) -> List[Tuple[str, float]]:
    return scores[:k]


def pick_top2_tags(
    scores: List[Tuple[str, float]],
    min_score: float = 0.35,
    max_tags: int = 2,
    within_margin: float = 0.03,      # accept if close to best
    ratio_of_best: float = 0.95,      # or if >= 95% of best
) -> List[Tuple[str, float]]:
    """
    Multi-label selection:
      - include best if >= min_score
      - include next tags if they are strong and near-tied with best
    """
    if not scores:
        return []

    best_tag, best_score = scores[0]
    if best_score < min_score:
        return []

    picked = [(best_tag, best_score)]

    for tag, s in scores[1:]:
        if len(picked) >= max_tags:
            break
        if s < min_score:
            break

        near_tie = (best_score - s) <= within_margin
        strong_ratio = s >= (best_score * ratio_of_best)

        if near_tie or strong_ratio:
            picked.append((tag, s))

    return picked

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    device = "cpu"  # or "cuda" if available
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    # Print tags found in your dataset
    print(get_all_tag_categories_from_jsonl("schemas/chunks.jsonl"))

    # Better synonym sets: avoid generic "main course/main" (it blurs classes).
    TAG_SYNONYMS: Dict[str, List[WeightedSyn]] = {
        "vegetarian_candidate": [
            ("vegetarian", 2.0),
            ("meatless", 1.5),
            ("no meat", 1.5),
            ("plant-based", 1.2),
            ("veggie", 1.0),
            ("vegetable dish", 1.0),
            ("without meat", 1.5),
        ],
        "dessert": [
            ("dessert", 2.0),
            ("sweet dish", 1.5),
            ("cake", 1.0),
            ("pudding", 1.0),
            ("treat", 0.8),
            ("after dinner sweet", 1.0),
        ],
        "meat": [
            ("meat", 2.0),
            ("beef", 1.0),
            ("pork", 1.0),
            ("chicken", 1.0),
            ("ham", 1.0),
            ("bacon", 1.0),
            ("minced meat", 0.8),
        ],
        "pasta": [
            ("pasta", 2.0),
            ("pasta dish", 1.4),
            ("spaghetti", 1.0),
            ("penne", 1.0),
            ("lasagna", 1.0),
            ("macaroni", 0.8),
            ("fettuccine", 0.8),
        ],
        "seafood": [
            ("seafood", 2.0),
            ("fish", 1.5),
            ("shellfish", 1.0),
            ("shrimp", 1.0),
            ("salmon", 1.0),
            ("tuna", 0.9),
            ("mussels", 0.8),
        ],
    }

    tag_vecs = build_tag_vectors(
        model,
        TAG_SYNONYMS,
        device=device,
        template="This recipe is {text}.",
    )

    queries = [
        "I want something without meat",
        "a sweet dish after dinner",
        "recipe with bacon and ham",
        "What are the ingredients to tiramisu?",
        "What are the ingredients to lasagna?",
    ]

    for q in queries:
        scores = score_query_against_tags(
            model, q, tag_vecs, device=device, template="User request: {text}"
        )
        print(f"\nQuery: {q}")
        for tag, s in top_k(scores, k=5):
            print(f"  {tag:22s}  {s:.3f}")

        picked = pick_top2_tags(scores, min_score=0.35, within_margin=0.03, ratio_of_best=0.95)
        print("  picked:", picked)      
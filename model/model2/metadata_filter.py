import json
from typing import Dict, List, Sequence, Set, Optional, Tuple, Union

import numpy as np
from sentence_transformers import SentenceTransformer


import re
from typing import Dict, List, Tuple

TagScores = List[Tuple[str, float]]

_NEGATION_PATTERNS: Dict[str, List[re.Pattern]] = {
    # If query indicates "no meat", penalize the "meat" tag
    "meat": [
        re.compile(r"\b(no|without|w\/o|not)\s+(any\s+)?meat\b", re.I),
        re.compile(r"\b(meat[-\s]?free|no\s+meat)\b", re.I),
        re.compile(r"\b(no|without|w\/o|not)\s+(any\s+)?(bacon|ham|beef|pork|chicken)\b", re.I),
        re.compile(r"\b(vegetarian|vegan)\b", re.I),  # optional: treat vegetarian/vegan as excluding meat
    ],
    # If query indicates "no seafood", penalize the "seafood" tag
    "seafood": [
        re.compile(r"\b(no|without|w\/o|not)\s+(any\s+)?(seafood|fish|shellfish)\b", re.I),
        re.compile(r"\b(seafood[-\s]?free|fish[-\s]?free)\b", re.I),
    ],
    # Optional examples:
    # "pasta": [re.compile(r"\b(gluten[-\s]?free|no\s+gluten)\b", re.I)],
}

def apply_negation_gate(
    query: str,
    scores: TagScores,
    penalty: float = 0.15,   # multiply forbidden tag score by this (0.0 blocks, 0.1..0.3 soft)
) -> TagScores:
    score_map = dict(scores)

    for tag, patterns in _NEGATION_PATTERNS.items():
        if tag not in score_map:
            continue
        if any(p.search(query) for p in patterns):
            score_map[tag] *= penalty

    gated = list(score_map.items())
    gated.sort(key=lambda x: x[1], reverse=True)
    return gated

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

def pick_top_k_with_margin(scores: TagScores, k: int = 2, margin: float = 0.03) -> TagScores:
    if not scores:
        return []
    picked = [scores[0]]
    for tag, s in scores[1:]:
        if len(picked) >= k:
            break
        if picked[0][1] - s <= margin:
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

    TAG_SYNONYMS: Dict[str, List[WeightedSyn]] = {
        "vegetarian_candidate": [
            ("vegetarian", 2.0),
            ("meatless", 1.5),
            ("no meat", 1.5),
            ("plant-based", 1.2),
            ("veggie", 1.0),
            ("vegetable dish", 1.0),
            # ("without meat", 1.5),
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
            ("with meat", 2.0),
            ("with beef", 1.0),
            ("with pork", 1.0),
            ("with chicken", 1.0),
            ("with ham", 1.0),
            ("with bacon", 1.0),
            ("minced meat", 0.8),
            ("contains meat", 1.5)
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
        "a sweet dish",
        "A carbonara without meat",
        "recipe with bacon and ham",
        "What are the ingredients to tiramisu?",
        "What are the ingredients to lasagna?",
    ]

    for q in queries:
        scores = score_query_against_tags(
            model, q, tag_vecs, device=device, template="User request: {text}"
        )

        scores = apply_negation_gate(q, scores, penalty=0.10)  # strong penalty
        # print(top_k(scores, k=5))
        print(f"\nQuery: {q}")
        for tag, s in top_k(scores, k=5):
            print(f"  {tag:22s}  {s:.3f}")

        picked = pick_top_k_with_margin(scores, k=2, margin=0.03)
        print("  picked:", picked)      
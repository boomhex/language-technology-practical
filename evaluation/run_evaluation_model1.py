"""
Run evaluation for model1 with retrieval + generation metrics.
Output JSON includes a `summary` section plus per-item results.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from model.generator import GenerativeQA
from model.similarity import build_similarity_index
from model.load import load_chunk_recipes
from model.question import ask_recipe_question


# Configuration 

GEN_MODEL_NAME = "google/flan-t5-base"
ENC_MODEL_NAME = "all-MiniLM-L6-v2"
DEVICE = "cpu"
CHUNK_TOKEN_SIZE = 400
RECIPES = 200

# Retrieval for the model answer (context size)
TOPK = 3

K_METRICS = 10

ROOT = Path(__file__).resolve().parents[1]  # the outer "model" dir
DATA_SRC = ROOT / "data"
EVAL_PATH = ROOT / "evaluation" / "evaluation_model1.json"
OUT_PATH = ROOT / "evaluation" / "eval_results_model1.json"


## Metrics  
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def token_f1(pred: str, gold: str) -> float:
    pt = _norm(pred).split()
    gt = _norm(gold).split()
    if not pt and not gt:
        return 1.0
    if not pt or not gt:
        return 0.0
    from collections import Counter

    pc, gc = Counter(pt), Counter(gt)
    common = sum((pc & gc).values())
    if common == 0:
        return 0.0
    prec = common / len(pt)
    rec = common / len(gt)
    return 2 * prec * rec / (prec + rec)


def list_f1(pred_items: Sequence[str], gold_items: Sequence[str]) -> float:
    p = {_norm(x) for x in pred_items if _norm(x)}
    g = {_norm(x) for x in gold_items if _norm(x)}
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    inter = len(p & g)
    prec = inter / len(p)
    rec = inter / len(g)
    return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)


def answer_accuracy(pred: str, expected: Any) -> float:
    """Heuristic accuracy in [0,1].

    - If expected is a list: parse list-like output and compute F1.
    - If expected is yes/no: check first token.
    - Else: substring match then fallback to token-F1.
    """

    pred_n = _norm(pred)

    if isinstance(expected, list):
        cand = [
            x.strip()
            for x in re.split(r"[\n,;•\-\*]+", pred)
            if x.strip()
        ]
        f1 = list_f1(cand, expected)
        return 1.0 if f1 >= 0.70 else float(f1)

    exp_n = _norm(str(expected))
    if exp_n in {"yes", "no"}:
        if pred_n.startswith("yes") or pred_n in {"y", "true"}:
            pred_yn = "yes"
        elif pred_n.startswith("no") or pred_n in {"n", "false"}:
            pred_yn = "no"
        else:
            pred_yn = pred_n.split(" ")[0] if pred_n else ""
        return 1.0 if pred_yn == exp_n else 0.0

    if exp_n and exp_n in pred_n:
        return 1.0

    f1 = token_f1(pred, str(expected))
    return 1.0 if f1 >= 0.80 else float(f1)


def mrr_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    gold_set = set(gold)
    for i, x in enumerate(ranked[:k], start=1):
        if x in gold_set:
            return 1.0 / i
    return 0.0


def precision_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    if k <= 0:
        return 0.0
    gold_set = set(gold)
    topk = ranked[:k]
    return float(sum(1 for x in topk if x in gold_set) / max(len(topk), 1))


def recall_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    topk = ranked[:k]
    return float(sum(1 for x in topk if x in gold_set) / len(gold_set))


def hit_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    gold_set = set(gold)
    return 1.0 if any(x in gold_set for x in ranked[:k]) else 0.0


def ndcg_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    gold_set = set(gold)
    if not gold_set:
        return 0.0

    def dcg(seq: Sequence[str]) -> float:
        score = 0.0
        for i, x in enumerate(seq[:k], start=1):
            if x in gold_set:
                score += 1.0 / math.log2(i + 1)
        return score

    dcg_val = dcg(ranked)
    ideal = dcg(list(gold_set))  # any order is fine for binary relevance
    return float(dcg_val / ideal) if ideal > 0 else 0.0


def cosine_sim(embed_model, a: str, b: str) -> float:
    """Cosine similarity between two strings using the existing encoder."""
    va = embed_model.encode([a], normalize_embeddings=True)[0]
    vb = embed_model.encode([b], normalize_embeddings=True)[0]
    return float(np.dot(va, vb))


def _aggregate(rows: List[Dict[str, float]], key: str) -> float:
    return float(np.mean([r.get(key, 0.0) for r in rows])) if rows else 0.0



# Gold id resolution

def _load_json_list(path: Path) -> List[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [str(x) for x in data]
    # allow {"ids": [...]} style
    if isinstance(data, dict):
        for k in ("ids", "doc_ids", "chunk_ids", "gold", "values"):
            if k in data and isinstance(data[k], list):
                return [str(x) for x in data[k]]
    return []


def resolve_gold_ids(
    item: Dict[str, Any],
    base_dir: Path,
    field: str,
    file_field: Optional[str] = None,
) -> List[str]:

    # explicit file field wins
    if file_field and item.get(file_field):
        p = base_dir / str(item[file_field])
        return _load_json_list(p) if p.exists() else []

    val = item.get(field)
    if val is None:
        return []

    # sometimes the field itself is a path string
    if isinstance(val, str):
        if val.upper() == "MISSING":
            return []
        p = base_dir / val
        if p.exists() and p.suffix.lower() == ".json":
            return _load_json_list(p)
        return [val]

    if isinstance(val, list):
        return [str(x) for x in val if str(x).upper() != "MISSING"]

    return [str(val)]


def _dedup_keep_first(xs: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _conversation_append(ctx: str, user_text: str, assistant_text: str) -> str:
    ctx = ctx or ""
    if ctx and not ctx.endswith("\n"):
        ctx += "\n"
    ctx += f"User: {user_text}\nAssistant: {assistant_text}\n"
    return ctx


def main() -> None:
    eval_data = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    questions = eval_data["questions"]
    base_dir = EVAL_PATH.parent

    recipe_chunks = load_chunk_recipes(
        data_dir=DATA_SRC,
        model_name=GEN_MODEL_NAME,
        nr_recipes=RECIPES,
        max_tokens=CHUNK_TOKEN_SIZE,
        overlap_tokens=0,
    )
    index, embed_model = build_similarity_index(
        recipe_chunks=recipe_chunks,
        model_name=ENC_MODEL_NAME,
        device=DEVICE,
        batch_size=32,
    )
    gen_qa = GenerativeQA(device=DEVICE, model_name=GEN_MODEL_NAME)

    results: List[Dict[str, Any]] = []

    doc_rows, chunk_rows, gen_rows = [], [], []

    def eval_single_question(
        q_text: str,
        conversation_context: str,
        expected_answer: Any = None,
        gold_doc_ids: Optional[List[str]] = None,
        gold_chunk_ids: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, Any], str]:
        answer, hits = ask_recipe_question(
            gen_qa=gen_qa,
            question=q_text,
            recipe_chunks=recipe_chunks,
            embed_model=embed_model,
            faiss_index=index,
            k_retrieval=TOPK,
            device=DEVICE,
            conversation_context=conversation_context or "",
        )

        ranked_doc_ids = [h["chunk"]["doc_id"] for h in hits]
        ranked_chunk_ids = [h["chunk_id"] for h in hits]
        ranked_doc_ids_dedup = _dedup_keep_first(ranked_doc_ids)

        k = K_METRICS

        doc_metrics = {}
        if gold_doc_ids:
            doc_metrics = {
                "mrr": mrr_at_k(ranked_doc_ids_dedup, gold_doc_ids, k),
                "precision": precision_at_k(ranked_doc_ids_dedup, gold_doc_ids, k),
                "recall": recall_at_k(ranked_doc_ids_dedup, gold_doc_ids, k),
                "hit": hit_at_k(ranked_doc_ids_dedup, gold_doc_ids, k),
                "ndcg": ndcg_at_k(ranked_doc_ids_dedup, gold_doc_ids, k),
            }
            doc_rows.append(doc_metrics)

        chunk_metrics = {}
        if gold_chunk_ids:
            chunk_metrics = {
                "mrr": mrr_at_k(ranked_chunk_ids, gold_chunk_ids, k),
                "precision": precision_at_k(ranked_chunk_ids, gold_chunk_ids, k),
                "recall": recall_at_k(ranked_chunk_ids, gold_chunk_ids, k),
                "hit": hit_at_k(ranked_chunk_ids, gold_chunk_ids, k),
                "ndcg": ndcg_at_k(ranked_chunk_ids, gold_chunk_ids, k),
            }
            chunk_rows.append(chunk_metrics)

        gen_metrics = {}
        if expected_answer is not None:
            acc = answer_accuracy(answer, expected_answer)
            f1 = acc if isinstance(expected_answer, list) else token_f1(answer, str(expected_answer))
            gold_text = (
                "; ".join(map(str, expected_answer))
                if isinstance(expected_answer, list)
                else str(expected_answer)
            )
            cos = cosine_sim(embed_model, answer, gold_text)
            gen_metrics = {"accuracy": acc, "f1": f1, "cosine_similarity": cos}
            gen_rows.append(gen_metrics)

        top_hits = [
            {
                "rank": h["rank"],
                "score": h["score"],
                "doc_id": h["chunk"]["doc_id"],
                "title": h["chunk"]["title"],
                "chunk_id": h["chunk_id"],
            }
            for h in hits
        ]

        out = {
            "question": q_text,
            "expected_answer": expected_answer,
            "model_answer": answer,
            "retrieval": {
                "k_metrics": k,
                "ranked_doc_ids": ranked_doc_ids,
                "ranked_chunk_ids": ranked_chunk_ids,
                "gold_doc_ids": gold_doc_ids or [],
                "gold_chunk_ids": gold_chunk_ids or [],
                "doc_metrics": doc_metrics,
                "chunk_metrics": chunk_metrics,
            },
            "gen_metrics": gen_metrics,
            "top_hits": top_hits,
        }

        # update multi-turn context
        new_ctx = _conversation_append(conversation_context, q_text, answer)
        return out, new_ctx

    for item in questions:
        item_id = item.get("id")
        item_type = str(item.get("type", "")).lower()

        if item_type == "multi-turn" or "turns" in item:
            parent_gold_docs = resolve_gold_ids(
                item, base_dir, field="gold_doc_id", file_field="gold_doc_id_file"
            )

            conversation_context = ""
            turns_out = []
            for t_i, turn in enumerate(item.get("turns", [])):
                # turns use either `text` or `question`
                q_text = turn.get("text") or turn.get("question")
                expected = turn.get("expected_answer")
                gold_chunks = resolve_gold_ids(turn, base_dir, field="gold_chunk_ids")

                turn_res, conversation_context = eval_single_question(
                    q_text=q_text,
                    conversation_context=conversation_context,
                    expected_answer=expected,
                    gold_doc_ids=parent_gold_docs,
                    gold_chunk_ids=gold_chunks,
                )
                turn_res["turn_index"] = t_i
                turns_out.append(turn_res)

                best = turn_res["top_hits"][0]["score"] if turn_res["top_hits"] else None
                print(f"[{item_id:02d}.{t_i}] multi-turn | best={best}")
                print("Q:", q_text)
                print("A:", turn_res["model_answer"])
                print("-" * 60)

            results.append({"id": item_id, "type": item.get("type"), "turns": turns_out})
            continue

        # single-turn
        q_text = item.get("question")
        expected = item.get("expected_answer")
        gold_docs = resolve_gold_ids(item, base_dir, field="gold_doc_id", file_field="gold_doc_id_file")
        gold_chunks = resolve_gold_ids(item, base_dir, field="gold_chunk_ids")

        res, _ = eval_single_question(
            q_text=q_text,
            conversation_context="",
            expected_answer=expected,
            gold_doc_ids=gold_docs,
            gold_chunk_ids=gold_chunks,
        )

        res.update(
            {
                "id": item_id,
                "type": item.get("type"),
                "expected_behavior": item.get("expected_behavior"),
            }
        )
        results.append(res)

        best = res["top_hits"][0]["score"] if res["top_hits"] else None
        print(f"[{item_id:02d}] {item.get('type')} | best={best}")
        print("Q:", q_text)
        print("A:", res["model_answer"])
        print("-" * 60)

    summary = {
        "k_retrieval": TOPK,
        "k_metrics": K_METRICS,
        "retrieval": {
            "doc": {
                "MRR@k": _aggregate(doc_rows, "mrr"),
                "Precision@k": _aggregate(doc_rows, "precision"),
                "Recall@k": _aggregate(doc_rows, "recall"),
                "Hit@k": _aggregate(doc_rows, "hit"),
                "nDCG@k": _aggregate(doc_rows, "ndcg"),
                "n_items": len(doc_rows),
            },
            "chunk": {
                "MRR@k": _aggregate(chunk_rows, "mrr"),
                "Precision@k": _aggregate(chunk_rows, "precision"),
                "Recall@k": _aggregate(chunk_rows, "recall"),
                "Hit@k": _aggregate(chunk_rows, "hit"),
                "nDCG@k": _aggregate(chunk_rows, "ndcg"),
                "n_items": len(chunk_rows),
            },
        },
        "generation": {
            "Accuracy": _aggregate(gen_rows, "accuracy"),
            "F1": _aggregate(gen_rows, "f1"),
            "CosineSimilarity": _aggregate(gen_rows, "cosine_similarity"),
            "n_items": len(gen_rows),
        },
    }

    OUT_PATH.write_text(
        json.dumps({"summary": summary, "results": results}, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved eval results to: {OUT_PATH}")


if __name__ == "__main__":
    main()

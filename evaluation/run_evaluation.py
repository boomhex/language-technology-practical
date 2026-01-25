"""
Error analysis script for week 3 error analysis
Help identify what elements need more attention 
"""

import json
from pathlib import Path

from model.generator import GenerativeQA
from model.similarity import build_similarity_index
from model.load import load_chunk_recipes
from model.question import ask_recipe_question

GEN_MODEL_NAME = "google/flan-t5-base"
ENC_MODEL_NAME = "all-MiniLM-L6-v2"
DEVICE = "cpu"
CHUNK_TOKEN_SIZE = 400
RECIPES = 12          # set to 200 when available 
TOPK = 3

ROOT = Path(__file__).resolve().parents[1]  # the outer "model" dir
DATA_SRC = ROOT / "data"
EVAL_PATH = ROOT / "evaluation" / "evaluation.json"
OUT_PATH = ROOT / "evaluation" / "eval_results_raw.json"

def main():
    eval_data = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    questions = eval_data["questions"]

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

    results = []
    for q in questions:
        answer, hits = ask_recipe_question(
            gen_qa=gen_qa,
            question=q["question"],
            recipe_chunks=recipe_chunks,
            embed_model=embed_model,
            faiss_index=index,
            k_retrieval=TOPK,
            device=DEVICE,
            conversation_context="",
        )

        best = hits[0]["score"] if hits else None
        top = [
            {
                "rank": h["rank"],
                "score": h["score"],
                "doc_id": h["chunk"]["doc_id"],
                "title": h["chunk"]["title"],
                "chunk_id": h["chunk_id"],
            }
            for h in hits
        ]

        results.append(
            {
                "id": q["id"],
                "type": q["type"],
                "question": q["question"],
                "expected_answer": q.get("expected_answer"),
                "expected_behavior": q.get("expected_behavior"),
                "model_answer": answer,
                "best_score": best,
                "top_hits": top,
            }
        )

        print(f"[{q['id']:02d}] {q['type']} | best={best}")
        print("Q:", q["question"])
        print("A:", answer)
        print("-" * 60)

    OUT_PATH.write_text(
    json.dumps({"results": results}, indent=2),
    encoding="utf-8"
    )
    print(f"\nSaved raw results to: {OUT_PATH}")


if __name__ == "__main__":
    main()
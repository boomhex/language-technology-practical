# test_classifier.py
from __future__ import annotations
import sys
from pathlib import Path

# add project root (parent of tests/) to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from query_classifier import classify_to_filter
from logic_gen import Phi3Caller

ALLOWED_SECTIONS = ["ingredients", "steps"]
ALLOWED_TAGS = ["vegetarian_candidate", "vegan", "gluten_free", "dairy_free", "eggless"]

def main() -> None:
    qa_model = Phi3Caller("cpu")
    test_queries = [
        "tiramisu",
        "tiramisu ingredients",
        "eggless tiramisu ingredients",
        "vegan tiramisu",
        "gluten-free tiramisu directions",
        "Roasted Pumpkin Cream ingredients vegetarian",
        "doc_029f7d15edfc9453 ingredients",
    ]

    for q in test_queries:
        flt = classify_to_filter(qa_model,q, allowed_sections=ALLOWED_SECTIONS, allowed_tags=ALLOWED_TAGS)
        print("=" * 80)
        print("Query:", q)
        print("MetaFilter:", flt)


if __name__ == "__main__":
    main()
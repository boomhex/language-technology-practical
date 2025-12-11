from generator import GenerativeQA
from similarity import build_similarity_index
from load import load_chunk_recipes
from question import ask_recipe_question

GEN_MODEL_NAME = "google/flan-t5-base"
ENC_MODEL_NAME = "all-MiniLM-L6-v2"
DEVICE = "cpu"
CHUNK_TOKEN_SIZE = 400
RECIPES = 1050
TOPK = 3
DATA_SRC = "../data"

def interactive_recipe_qa(
    gen_qa,
    recipe_chunks,
    embed_model,
    faiss_index,
    k_retrieval,
):
    print("Recipe QA – ask a question about your recipes.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        q = input("Ask a question about your recipes (or type 'quit'): ").strip()

        if q.lower() in {"quit", "exit"}:
            print("Exiting Recipe QA.")
            break

        if not q:
            print("Please enter a non-empty question.\n")
            continue

        # If you want normalization, uncomment:
        # q = normalize_text(q)

        answer, hits = ask_recipe_question(
            gen_qa,
            question=q,
            recipe_chunks=recipe_chunks,
            embed_model=embed_model,
            faiss_index=faiss_index,
            k_retrieval=k_retrieval,
        )

        print("\nQuestion:")
        print(q)
        print("\nAnswer:")
        print(answer)

        print("\nTop retrieved recipe chunks:")
        for h in hits[:10]:
            print(f"- Recipe {h['chunk']['doc_id']} ({h['chunk']['title']}), "
                  f"score={h['score']:.3f}")
        print("-" * 60)
        print()  # extra newline for readability


def main():
    recipe_chunks = load_chunk_recipes(data_dir=DATA_SRC, 
                                       model_name=GEN_MODEL_NAME, 
                                       nr_recipes=RECIPES,
                                       max_tokens=CHUNK_TOKEN_SIZE )
    index, embed_model= build_similarity_index(recipe_chunks=recipe_chunks, 
                                   model_name=ENC_MODEL_NAME, 
                                   device=DEVICE, 
                                   batch_size=32)
    gen_qa = GenerativeQA(device=DEVICE, model_name=GEN_MODEL_NAME)

    interactive_recipe_qa(gen_qa=gen_qa, 
                          recipe_chunks=recipe_chunks, 
                          embed_model=embed_model, 
                          faiss_index=index, 
                          k_retrieval=TOPK)

    

if __name__ == "__main__":
    main()
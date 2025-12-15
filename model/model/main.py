from generator import GenerativeQA
from similarity import build_similarity_index
from load import load_chunk_recipes
from question import ask_recipe_question
from memory import ShortTermMemory

short_memory = ShortTermMemory(max_size=12)

GEN_MODEL_NAME = "google/flan-t5-base"
ENC_MODEL_NAME = "all-MiniLM-L6-v2"
DEVICE = "cpu"
CHUNK_TOKEN_SIZE = 400
RECIPES = 12
TOPK = 3
DATA_SRC = "../data"

def handle_question(
    user_input: str,
    gen_qa,
    recipe_chunks,
    embed_model,
    faiss_index,
    k_retrieval: int,
    similarity_threshold: float = 0.2,
):
    """
    Single QA turn with short-term memory only.

    - Store the user question in short_memory
    - Call existing ask_recipe_question()
    - Look at retrieval scores; if best score is too low,
      respond with a clarification question instead.
    - Store the assistant reply in short_memory as well.
    """
    # 1. remember what the user said
    short_memory.add("user", user_input)

    # 2. normal retrieval + generation
    context_history = short_memory.get_context()
    print(context_history)
    query = gen_qa.rewrite(user_input, context_history)
    print(f"new query: {query}\n")

    answer, hits = ask_recipe_question(
        gen_qa=gen_qa,
        question=query,
        recipe_chunks=recipe_chunks,
        embed_model=embed_model,
        faiss_index=faiss_index,
        k_retrieval=k_retrieval,
        conversation_context=context_history,     # NEW
    )

    # 3. decide if we are confident enough, based on retrieval scores
    best_score = hits[0]["score"] if hits else 0.0

    if best_score < similarity_threshold:
        clarification = (
            "I am not confident I have enough relevant recipe information to "
            "answer that. Could you rephrase the question or be more specific, "
            "for example by mentioning a dish or ingredient?"
        )
        short_memory.add("assistant", clarification)
        return clarification, hits

    # 4. otherwise, use the generated answer
    short_memory.add("assistant", answer)
    return answer, hits


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

        # >>> use short-memory-aware handler instead of calling ask_recipe_question directly
        answer, hits = handle_question(
            user_input=q,
            gen_qa=gen_qa,
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
            print(
                f"- Recipe {h['chunk']['doc_id']} "
                f"({h['chunk']['title']}), score={h['score']:.3f}"
            )
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
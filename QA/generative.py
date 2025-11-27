from pathlib import Path
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    pipeline,
)
from utils import (
    max_over_ground_truths,
    compute_f1,
    read_file,
    load_questions,
    compute_exact_match
)

GEN_MODEL_NAME = "t5-small"
DATA_FOLDER = Path( "./../data" )
FILES = [DATA_FOLDER.joinpath(Path( str(id) + "/data.txt" )) for id in range(5)]
MAX_LEN = 384 # feel free to change
N_EVAL_EXAMPLES = 300  # using a subset of the dataset, feel free to change


def evaluate_generative(context, questions) -> tuple[float, float]:
    """
    Evaluate a generative QA model:
    - Format input as: "question: ... context: ..."
    - Generate an answer string
    - Compare to ground truth with EM/F1
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Generative] Using device: {device}")

    gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME).to(device)

    gen_pipe = pipeline(
        "text2text-generation",
        model=gen_model,
        tokenizer=gen_tokenizer,
        device=0 if device.type == "cuda" else -1,
    )

    em_scores = []
    f1_scores = []

    for el in questions.values():
        question = el["question"]
        context = context
        ground_truths = el["answer"]

        # Prompt format for the model
        prompt = f"question: {question}  context: {context}"

        try:
            out = gen_pipe(
                prompt,
                max_new_tokens=32, # max answer length
                num_beams=4, # 4 candidates
                do_sample=False, #deterministic
            )[0]["generated_text"]
            prediction_text = out.strip()
        except Exception as e:
            prediction_text = ""

        print(prediction_text, ground_truths)

        em = max_over_ground_truths(compute_exact_match, prediction_text, ground_truths)
        f1 = max_over_ground_truths(compute_f1, prediction_text, ground_truths)

        em_scores.append(em)
        f1_scores.append(f1)

    avg_em = 100.0 * sum(em_scores) / len(em_scores)
    avg_f1 = 100.0 * sum(f1_scores) / len(f1_scores)

    return avg_em, avg_f1


def main() -> None:
    context = "\n".join(
        [read_file(file) for file in FILES]
    )   # concatenate context
    questions_file = Path( "./../questions/questions.json" )
    questions = load_questions(questions_file)
    em, f1 = evaluate_generative(context, questions)
    print(f"em: {em}, f1: {f1}")

if __name__ == "__main__":
    main()
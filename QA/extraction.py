from utils import (
    read_file,
    compute_f1,
    compute_exact_match,
    normalize_answer,
    max_over_ground_truths,
    load_questions
)
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    pipeline,
)
import torch
from collections import Counter
import re
import json

EXTRACTIVE_MODEL_NAME = "distilbert-base-uncased-distilled-squad"
DATA_FOLDER = Path( "./../data" )
FILES = [DATA_FOLDER.joinpath(Path( str(id) + "/data.txt" )) for id in range(1)]
MAX_LEN = 384 # feel free to change
N_EVAL_EXAMPLES = 300  # using a subset of the dataset, feel free to change

def evaluate_extractive(context, questions):
    """
    Evaluate a span-extractive QA model:
    - Uses AutoModelForQuestionAnswering
    - Manually tokenizes question+context
    - Picks best start/end tokens from logits
    - Maps back to answer text using offset mapping
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Extractive] Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(EXTRACTIVE_MODEL_NAME, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(EXTRACTIVE_MODEL_NAME).to(device)
    model.eval()

    em_scores = []
    f1_scores = []

    for el in questions.values():
        # question = ex["question"].strip()
        context = context
        ground_truths = el["answer"]
        question = el["question"]

        # Tokenize question + context.
        # return_offsets_mapping=True saves token position in the original context
        encoded = tokenizer(
            question,
            context,
            max_length=MAX_LEN,
            truncation="only_second",
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        # sequence_ids tells us which tokens belong to question (0) and context (1)
        sequence_ids = encoded.sequence_ids(0)
        offset_mapping = encoded["offset_mapping"][0]

        # Remove offset_mapping from inputs before sending to model
        encoded_without_offsets = {
            key: value.to(device)
            for key, value in encoded.items()
            if key != "offset_mapping"
        }

        with torch.no_grad():
            outputs = model(**encoded_without_offsets)

        #Compute start and end logits
        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]

        # We only consider tokens that belong to the context
        context_indices = [idx for idx, sid in enumerate(sequence_ids) if sid == 1]

        if not context_indices:
            prediction_text = ""
        else:
            start_index = max(context_indices, key=lambda idx: start_logits[idx].item())
            end_index = max(context_indices, key=lambda idx: end_logits[idx].item())

            if end_index < start_index:
                # Invalid span, give empty prediction
                prediction_text = ""
            else:
                start_char = offset_mapping[start_index][0].item()
                end_char = offset_mapping[end_index][1].item()
                prediction_text = context[start_char:end_char]
        print("prediction:\n")
        print(prediction_text)

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
    em, f1 = evaluate_extractive(context, questions)
    print(f"em: {em}, f1: {f1}")


if __name__ == "__main__":
    main()


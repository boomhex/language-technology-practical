"""
Week 2 QA Tutorial Script
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    pipeline,
)
from datasets import load_dataset
from collections import Counter
import string
import re
import random


# QA finetuned model:
EXTRACTIVE_MODEL_NAME = "distilbert-base-uncased-distilled-squad"
# Generative model:
GEN_MODEL_NAME = "t5-small"
MAX_LEN = 384 # feel free to change
N_EVAL_EXAMPLES = 300  # using a subset of the dataset, feel free to change



def normalize_answer(s: str) -> str:
    """
    Text normalization:
    - lowercase
    - remove punctuation
    - remove articles
    - collapse whitespace
    """

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """EM = 1 if normalized strings match exactly, else 0."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """
    Compute F1 on normalised tokens
    """
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    # check if one or both are empty, sanity check
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return float(pred_tokens == truth_tokens)

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def max_over_ground_truths(metric_fn, prediction: str, ground_truths):
    """
    If there are multiple ground truth answers,
    we take the maximum score over them.
    """
    return max(metric_fn(prediction, gt) for gt in ground_truths)



def load_squad_subset(n_examples: int = N_EVAL_EXAMPLES):
    """
    Load SQuAD validation split
    """
    squad = load_dataset("squad")
    val = squad["validation"]

    if n_examples is not None and n_examples < len(val):
        # set seed for reproducibility
        indices = list(range(len(val)))
        random.seed(42)
        random.shuffle(indices)
        indices = indices[:n_examples]
        val = val.select(indices)

    return val




def evaluate_extractive(dataset):
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

    for i, ex in enumerate(dataset):
        question = ex["question"].strip()
        context = ex["context"]
        ground_truths = ex["answers"]["text"]

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
            k: v.to(device)
            for k, v in encoded.items()
            if k != "offset_mapping"
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

        em = max_over_ground_truths(compute_exact_match, prediction_text, ground_truths)
        f1 = max_over_ground_truths(compute_f1, prediction_text, ground_truths)

        em_scores.append(em)
        f1_scores.append(f1)

        if (i + 1) % 50 == 0:
            print(f"[Extractive] Processed {i+1}/{len(dataset)} examples")

    avg_em = 100.0 * sum(em_scores) / len(em_scores)
    avg_f1 = 100.0 * sum(f1_scores) / len(f1_scores)

    return avg_em, avg_f1


# Generative model (T5)

def evaluate_generative(dataset):
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

    for i, ex in enumerate(dataset):
        question = ex["question"].strip()
        context = ex["context"]
        ground_truths = ex["answers"]["text"]

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

        em = max_over_ground_truths(compute_exact_match, prediction_text, ground_truths)
        f1 = max_over_ground_truths(compute_f1, prediction_text, ground_truths)

        em_scores.append(em)
        f1_scores.append(f1)

        if (i + 1) % 50 == 0:
            print(f"[Generative] Processed {i+1}/{len(dataset)} examples")

    avg_em = 100.0 * sum(em_scores) / len(em_scores)
    avg_f1 = 100.0 * sum(f1_scores) / len(f1_scores)

    return avg_em, avg_f1



def qualitative_examples(dataset, n_examples=1, max_context_chars=400):
    """
    Print a few examples comparing:
    - Context (optionally truncated)
    - Question
    - Gold answers
    - Extractive prediction
    - Generative prediction
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extractive components
    ext_tokenizer = AutoTokenizer.from_pretrained(EXTRACTIVE_MODEL_NAME, use_fast=True)
    ext_model = AutoModelForQuestionAnswering.from_pretrained(EXTRACTIVE_MODEL_NAME).to(device)
    ext_model.eval()

    # Generative components
    gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME).to(device)
    gen_pipe = pipeline(
        "text2text-generation",
        model=gen_model,
        tokenizer=gen_tokenizer,
        device=0 if device.type == "cuda" else -1,
    )

    print("\n================ QUALITATIVE EXAMPLES ================\n")

    # Select random examples
    indices = list(range(len(dataset)))
    random.seed(123)
    random.shuffle(indices)
    indices = indices[:n_examples]

    for idx in indices:
        ex = dataset[idx]
        question = ex["question"].strip()
        context = ex["context"]
        gold_answers = ex["answers"]["text"]

        # Optional context truncation for printing
        if max_context_chars and len(context) > max_context_chars:
            printable_context = context[:max_context_chars] + " ...[TRUNCATED]..."
        else:
            printable_context = context

        # Extractive prediction
        encoded = ext_tokenizer(
            question,
            context,
            max_length=MAX_LEN,
            truncation="only_second",
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        sequence_ids = encoded.sequence_ids(0)
        offsets = encoded["offset_mapping"][0]
        inputs = {k: v.to(device) for k, v in encoded.items() if k != "offset_mapping"}

        with torch.no_grad():
            outputs = ext_model(**inputs)

        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]
        context_indices = [i for i, sid in enumerate(sequence_ids) if sid == 1]

        # Extract best span
        if context_indices:
            start_index = max(context_indices, key=lambda i: start_logits[i].item())
            end_index = max(context_indices, key=lambda i: end_logits[i].item())
            if end_index >= start_index:
                start_char = offsets[start_index][0].item()
                end_char = offsets[end_index][1].item()
                ext_answer = context[start_char:end_char]
            else:
                ext_answer = ""
        else:
            ext_answer = ""

        # Generative prediction
        prompt = f"question: {question}  context: {context}"
        try:
            gen_out = gen_pipe(
                prompt,
                max_new_tokens=32,
                num_beams=4,
                do_sample=False,
            )[0]["generated_text"]
            gen_answer = gen_out.strip()
        except Exception:
            gen_answer = ""

        # Print
        print(f"--- Example {idx} ---\n")
        print("CONTEXT:")
        print(printable_context)
        print("\nQUESTION:")
        print(question)
        print("\nGOLD ANSWERS:")
        print(gold_answers)
        print("\nEXTRACTIVE PREDICTION:")
        print(ext_answer)
        print("\nGENERATIVE PREDICTION:")
        print(gen_answer)
        print("\n" + "="*60 + "\n")





def main():
    print("Loading SQuAD v1.1 validation subset...")
    dataset = load_squad_subset(N_EVAL_EXAMPLES)
    print(f"Loaded {len(dataset)} validation examples for evaluation.\n")

    # Extractive
    ext_em, ext_f1 = evaluate_extractive(dataset)
    print("\n==== Extractive QA (span-based) ====")
    print(f"Model: {EXTRACTIVE_MODEL_NAME}")
    print(f"EM: {ext_em:.2f}%")
    print(f"F1: {ext_f1:.2f}%")

    # Generative
    gen_em, gen_f1 = evaluate_generative(dataset)
    print("\n==== Generative QA (T5) ====")
    print(f"Model: {GEN_MODEL_NAME}")
    print(f"EM: {gen_em:.2f}%")
    print(f"F1: {gen_f1:.2f}%")

    # Predictions for inspection, feel free to change number or comment
    qualitative_examples(dataset, n_examples=5)


if __name__ == "__main__":
    main()

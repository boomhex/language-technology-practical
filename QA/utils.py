from pathlib import Path
import re
import string
from collections import Counter

def read_file(fp: Path) -> str:
    text = ""
    with open(fp, 'r') as file:
        text = file.read()
    return text


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



if __name__ == "__main__":
    fp = Path("./../data/0/data.txt")
    text = read_file(fp)
    print(text)

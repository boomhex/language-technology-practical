from pathlib import Path
import string
import random
from transformers import AutoTokenizer
import os
from typing import List, Dict
from typing import List
from transformers import PreTrainedTokenizerBase
import random

def remove_interpunction(text: str) -> str:
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)

def fix_whitespace(text: str) -> str:
    return ' '.join(text.split())

def read_file(path: Path) -> str:
    with open(path, 'r') as file:
        content = file.read()
    return content

def normalize_text(text: str) -> str:
    text = remove_interpunction(text)
    text = fix_whitespace(text)
    text = text.lower()
    return text

def load_recipes(base_dir: str = "data", max_index: int = 100) -> List[Dict]:
    """
    Load recipes from base_dir/i/data.txt for i in [0, max_index].
    Each loaded recipe is a dict with an id, title and text.
    """
    docs = []
    for i in range(max_index):  # 0..200
        recipe_path = os.path.join(base_dir, f"{i}.txt")

        if not os.path.exists(recipe_path):
            # Skip missing indices gracefully
            print(f"Warning: {recipe_path} not found, skipping.")
            continue

        with open(recipe_path, "r", encoding="utf-8") as f:
            text = f.read()
            text = normalize_text(text)

        docs.append(
            {
                "id": str(i),
                "title": f"recipe_{i}",
                "text": text,
                "path": recipe_path,
            }
        )
    return docs


def chunk_text_tokens(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int = 200,
    overlap_tokens: int = 50,
) -> List[str]:
    """
    Chunk `text` into overlapping windows based on *token* count.

    - `max_tokens`: maximum number of tokens per chunk.
    - `overlap_tokens`: number of tokens to overlap between consecutive chunks.

    Returns a list of text chunks (substrings of `text`), each corresponding
    to a contiguous span of tokens.
    """

    # Temporarily increase model_max_length so the tokenizer doesn't complain
    old_max_len = tokenizer.model_max_length
    tokenizer.model_max_length = int(1e6)  # something large

    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,  # we WANT the full sequence here
        )
    finally:
        # Restore original max length
        tokenizer.model_max_length = old_max_len

    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    n_tokens = len(input_ids)

    chunks = []
    start_tok = 0

    while start_tok < n_tokens:
        # End token index (exclusive)
        end_tok = min(n_tokens, start_tok + max_tokens)

        # Map token span back to character span
        start_char = offsets[start_tok][0]
        end_char = offsets[end_tok - 1][1]  # end index is exclusive

        chunk_text = text[start_char:end_char]
        chunks.append(chunk_text)

        if end_tok >= n_tokens:
            break

        # Next window with overlap in token space
        next_start_tok = end_tok - overlap_tokens

        # Safety: ensure progress
        if next_start_tok <= start_tok:
            next_start_tok = end_tok

        start_tok = next_start_tok

    return chunks

def build_recipe_chunks(recipes, tokenizer, max_tokens: int = 100, overlap_tokens: int = 50):
    """
    For each recipe document, create smaller token-based chunks.
    """
    chunks = []
    for doc in recipes:
        text = doc["text"]
        token_chunks = chunk_text_tokens(
            text,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
        for i, chunk in enumerate(token_chunks):
            chunks.append(
                {
                    "doc_id": doc["id"],
                    "chunk_id": f'{doc["id"]}_chunk_{i}',
                    "title": doc["title"],
                    "text": chunk,
                }
            )
    return chunks

def load_chunk_recipes(
    data_dir: str = "../model/data",
    model_name: str = "google/flan-t5-small",
    max_tokens: int = 400,
    overlap_tokens: int = 0,
):
    recipes = load_recipes(data_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    recipe_chunks = build_recipe_chunks(recipes, tokenizer, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    random.seed(123)
    random.shuffle(recipe_chunks)
    return recipe_chunks
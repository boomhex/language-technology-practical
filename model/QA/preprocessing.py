from pathlib import Path
import string

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

text = read_file("../data/1.txt")
print(normalize_text(text))
text = normalize_text(text)
with open("../data/aa_test.txt", 'w') as file:
    file.write(text)
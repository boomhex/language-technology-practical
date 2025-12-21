#!/usr/bin/env python3
"""
Build a normalized RAG schema + chunk records from semi-structured recipe text.

Input format (per document) like:
name:
...
ingredients:
...
info:
...
instructions:
...

Outputs:
- documents.jsonl  (one normalized document per line)
- chunks.jsonl     (section-aware chunks for retrieval)

Usage:
  python make_schema.py --input recipe.txt --out_dir out
  python make_schema.py --input_dir recipes_txt/ --out_dir out
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SECTION_ORDER = ["name", "ingredients", "info", "instructions"]


def stable_id(text: str, prefix: str = "doc") -> str:
    h = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"{prefix}_{h}"


def normalize_whitespace(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def split_sections(raw: str) -> Dict[str, str]:
    """
    Splits raw text into sections keyed by: name, ingredients, info, instructions.
    Tolerant to capitalization and extra whitespace.
    """
    raw = normalize_whitespace(raw)

    # Find section headers like "name:" at line start
    pattern = re.compile(r"^(name|ingredients|info|instructions)\s*:\s*$",
                         re.IGNORECASE | re.MULTILINE)

    matches = list(pattern.finditer(raw))
    if not matches:
        # If no headers found, treat entire thing as instructions (fallback)
        return {"name": "", "ingredients": "", "info": "", "instructions": raw}

    sections: Dict[str, str] = {k: "" for k in SECTION_ORDER}

    for i, m in enumerate(matches):
        key = m.group(1).lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        content = raw[start:end].strip()
        sections[key] = content

    return sections


def parse_minutes(value: str) -> Optional[int]:
    """
    Convert time strings like '60 min', '1 h 15 min', '45 mins' into minutes.
    Returns None if not parseable.
    """
    v = value.strip().lower()

    # Common patterns
    # 1) "60 min" / "60 mins"
    m = re.search(r"(\d+)\s*(min|mins|minute|minutes)\b", v)
    # 2) "1 h 15 min"
    h = re.search(r"(\d+)\s*(h|hr|hrs|hour|hours)\b", v)

    if not m and not h:
        return None

    minutes = 0
    if h:
        minutes += int(h.group(1)) * 60
    if m:
        minutes += int(m.group(1))
    return minutes if minutes > 0 else None


def parse_servings(value: str) -> Optional[int]:
    v = value.strip()
    m = re.search(r"(\d+)", v)
    return int(m.group(1)) if m else None


def parse_info_block(info_text: str) -> Dict[str, Any]:
    """
    Parses lines like:
      Difficulty: Average
      Prep time: 60 min
      Cook time: 45 min
      Serving: 8
      Cost: Average
      Note + chilling time...
    """
    out: Dict[str, Any] = {
        "difficulty": None,
        "cost": None,
        "prep_time": None,
        "cook_time": None,
        "total_time": None,
        "servings": None,
        "notes": []
    }

    lines = [ln.strip() for ln in info_text.split("\n") if ln.strip()]
    for ln in lines:
        # Key: Value
        kv = re.match(r"^([A-Za-z \-\+]+)\s*:\s*(.+)$", ln)
        if kv:
            key = kv.group(1).strip().lower()
            val = kv.group(2).strip()

            if key in {"difficulty"}:
                out["difficulty"] = val
            elif key in {"cost"}:
                out["cost"] = val
            elif key in {"prep time", "prep_time"}:
                out["prep_time"] = parse_minutes(val)
            elif key in {"cook time", "cook_time"}:
                out["cook_time"] = parse_minutes(val)
            elif key in {"serving", "servings", "yield"}:
                out["servings"] = parse_servings(val)
            else:
                # Preserve unknown structured lines as notes
                out["notes"].append(ln)
        else:
            # Free-form note line
            out["notes"].append(ln)

    # Compute total_time if possible
    if out["prep_time"] is not None or out["cook_time"] is not None:
        out["total_time"] = (out["prep_time"] or 0) + (out["cook_time"] or 0)

    return out


ING_LINE_RE = re.compile(r"^\s*(.+?)\s*:\s*(.+?)\s*$")


def parse_ingredients_block(ingredients_text: str) -> List[Dict[str, str]]:
    """
    Parses ingredient lines like:
      Type 00 flour: 3cups(380 g)
      Nutmeg: to taste
    """
    items: List[Dict[str, str]] = []
    for ln in ingredients_text.split("\n"):
        ln = ln.strip()
        if not ln:
            continue
        m = ING_LINE_RE.match(ln)
        if m:
            name = m.group(1).strip()
            amount = m.group(2).strip()
        else:
            # If formatting is odd, store whole line as name
            name, amount = ln, ""
        items.append({"name": name, "amount": amount})
    return items


def split_instructions(instructions_text: str) -> List[str]:
    """
    Splits instructions into steps. Since your input is paragraph-like,
    we:
      1) split by newline when present,
      2) otherwise split by sentence boundaries as a fallback.
    """
    text = normalize_whitespace(instructions_text)

    # Prefer newline-separated steps if they exist
    if "\n" in instructions_text:
        parts = [normalize_whitespace(p) for p in instructions_text.split("\n") if p.strip()]
        # Merge very short fragments into previous step
        steps: List[str] = []
        for p in parts:
            if steps and len(p) < 40:
                steps[-1] = (steps[-1] + " " + p).strip()
            else:
                steps.append(p)
        return steps

    # Fallback: sentence split (simple heuristic)
    sents = re.split(r"(?<=[.!?])\s+", text)
    sents = [s.strip() for s in sents if s.strip()]
    # Group into ~2-3 sentences per step
    steps: List[str] = []
    buf: List[str] = []
    for s in sents:
        buf.append(s)
        if len(buf) >= 3:
            steps.append(" ".join(buf))
            buf = []
    if buf:
        steps.append(" ".join(buf))
    return steps


def infer_tags(dish_name: str, ingredients: List[Dict[str, str]], instructions: str) -> List[str]:
    """
    Light heuristic tags. Keep it simple; you can upgrade later.
    """
    name_l = (dish_name or "").lower()
    ing_l = " ".join([i["name"].lower() for i in ingredients])
    txt = (name_l + " " + ing_l + " " + instructions.lower())

    tags = set()

    if any(k in txt for k in ["tart", "cake", "cookie", "dessert", "chocolate", "sugar", "icing", "ganache"]):
        tags.add("dessert")
    if any(k in txt for k in ["pasta", "spaghetti", "penne", "rigatoni", "tagliatelle"]):
        tags.add("pasta")
    if any(k in txt for k in ["fish", "salmon", "tuna", "anchovy", "shrimp", "prawn", "octopus", "mussel"]):
        tags.add("seafood")
    if any(k in txt for k in ["beef", "pork", "chicken", "lamb", "sausage", "guanciale", "bacon"]):
        tags.add("meat")
    # Very rough vegetarian guess
    if "meat" not in tags and "seafood" not in tags:
        tags.add("vegetarian_candidate")

    return sorted(tags)


def build_document_schema(raw: str, source: str, url: Optional[str] = None) -> Dict[str, Any]:
    sections = split_sections(raw)

    dish_name = sections.get("name", "").strip()
    ingredients = parse_ingredients_block(sections.get("ingredients", ""))
    info = parse_info_block(sections.get("info", ""))
    steps = split_instructions(sections.get("instructions", ""))

    doc_id = stable_id(dish_name + "\n" + raw, prefix="doc")

    # Normalize into your requested schema
    doc: Dict[str, Any] = {
        "doc_id": doc_id,
        "source": source,
        "url": url,
        "dish_name": dish_name or None,
        "region": None,  # fill later if you add region detection
        "doc_type": "recipe",
        "sections": {
            "ingredients": [f'{x["name"]}: {x["amount"]}'.strip(": ").strip() for x in ingredients],
            "steps": steps,
            "notes": "\n".join(info.get("notes", [])).strip() or None,
            "history": None,
            "variations": None,
            "techniques": None,
        },
        "servings": info.get("servings"),
        "prep_time": info.get("prep_time"),
        "cook_time": info.get("cook_time"),
        "total_time": info.get("total_time"),
        "tags": infer_tags(dish_name, ingredients, sections.get("instructions", "")),
        "parsed_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    return doc


def make_chunks(doc: Dict[str, Any], max_chars: int = 1200) -> List[Dict[str, Any]]:
    """
    Create section-aware chunks suitable for hybrid retrieval.
    Uses a simple char budget (works fine; you can switch to token budgets later).
    """
    chunks: List[Dict[str, Any]] = []

    def add_chunk(section: str, text: str, idx: int, extra: Optional[Dict[str, Any]] = None):
        c = {
            "chunk_id": f'{doc["doc_id"]}#{section}#{idx}',
            "doc_id": doc["doc_id"],
            "dish_name": doc.get("dish_name"),
            "region": doc.get("region"),
            "doc_type": doc.get("doc_type"),
            "source": doc.get("source"),
            "url": doc.get("url"),
            "section": section,
            "tags": doc.get("tags", []),
            "text": text.strip(),
        }
        if extra:
            c.update(extra)
        chunks.append(c)

    # Ingredients: usually one chunk
    ing = doc["sections"].get("ingredients") or []
    if ing:
        ing_text = "Ingredients:\n" + "\n".join(f"- {x}" for x in ing)
        add_chunk("ingredients", ing_text, 0)

    # Notes: one chunk
    notes = doc["sections"].get("notes")
    if notes:
        add_chunk("notes", "Notes:\n" + notes, 0)

    # Steps: chunk in groups
    steps = doc["sections"].get("steps") or []
    if steps:
        buf: List[str] = []
        idx = 0
        for i, step in enumerate(steps, start=1):
            candidate = "\n".join(buf + [f"{i}. {step}"]).strip()
            if buf and len(candidate) > max_chars:
                add_chunk("steps", "Instructions:\n" + "\n".join(buf), idx, extra={"step_start": i - len(buf), "step_end": i - 1})
                idx += 1
                buf = [f"{i}. {step}"]
            else:
                buf.append(f"{i}. {step}")

        if buf:
            start = int(re.match(r"^(\d+)\.", buf[0]).group(1)) if re.match(r"^(\d+)\.", buf[0]) else None
            end = int(re.match(r"^(\d+)\.", buf[-1]).group(1)) if re.match(r"^(\d+)\.", buf[-1]) else None
            add_chunk("steps", "Instructions:\n" + "\n".join(buf), idx, extra={"step_start": start, "step_end": end})

    # Times/servings as a compact factual chunk (very helpful for retrieval)
    facts = []
    if doc.get("servings") is not None:
        facts.append(f"Servings: {doc['servings']}")
    if doc.get("prep_time") is not None:
        facts.append(f"Prep time (min): {doc['prep_time']}")
    if doc.get("cook_time") is not None:
        facts.append(f"Cook time (min): {doc['cook_time']}")
    if doc.get("total_time") is not None:
        facts.append(f"Total time (min): {doc['total_time']}")
    if facts:
        add_chunk("info", "Info:\n" + "\n".join(f"- {x}" for x in facts), 0)

    return chunks


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=None, help="Single recipe text file")
    ap.add_argument("--input_dir", type=str, default=None, help="Directory of .txt files")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory")
    ap.add_argument("--source", type=str, default="local_text", help="Source label")
    ap.add_argument("--max_chunk_chars", type=int, default=1200, help="Approx chunk size")
    args = ap.parse_args()

    if not args.input and not args.input_dir:
        raise SystemExit("Provide --input or --input_dir")

    paths: List[Path] = []
    if args.input:
        paths.append(Path(args.input))
    if args.input_dir:
        d = Path(args.input_dir)
        paths.extend(sorted([p for p in d.glob("**/*.txt") if p.is_file()]))

    documents: List[Dict[str, Any]] = []
    chunks: List[Dict[str, Any]] = []

    for p in paths:
        raw = read_text_file(p)
        doc = build_document_schema(raw=raw, source=args.source, url=None)
        # Keep a pointer to the original filename (useful for debugging)
        doc["source_file"] = str(p)

        documents.append(doc)
        chunks.extend(make_chunks(doc, max_chars=args.max_chunk_chars))

    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "documents.jsonl", documents)
    write_jsonl(out_dir / "chunks.jsonl", chunks)

    print(f"Wrote {len(documents)} documents to {out_dir / 'documents.jsonl'}")
    print(f"Wrote {len(chunks)} chunks to {out_dir / 'chunks.jsonl'}")


if __name__ == "__main__":
    main()
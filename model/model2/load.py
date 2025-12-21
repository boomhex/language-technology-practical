from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

@dataclass(frozen=True)
class Chunk:
    text: str
    metadata: Dict[str, Any]

def _coerce_metadata(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accepts either:
      (A) {"text": "...", "metadata": {...}}
      (B) flat: {"text": "...", "chunk_id": "...", "doc_id": "...", ...}
    Returns metadata dict.
    """
    if "metadata" in obj and isinstance(obj["metadata"], dict):
        md = dict(obj["metadata"])
        # keep any other top-level fields (except text/metadata) as well
        for k, v in obj.items():
            if k not in {"text", "metadata"} and k not in md:
                md[k] = v
        return md

    # flat case: everything except text is metadata
    md = {k: v for k, v in obj.items() if k != "text"}
    return md

def load_chunks(data_path: str, limit: Optional[int] = None) -> List[Chunk]:
    """
    Loads JSONL where each line is a chunk-like object.
    Works with:
      - flat chunk lines (your example)
      - lines with {"text": "...", "metadata": {...}}
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() != ".jsonl":
        raise ValueError("This loader expects a .jsonl file.")

    out: List[Chunk] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and len(out) >= int(limit):
                break

            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue

            text = (obj.get("text") or "").strip()
            if not text:
                continue

            md = _coerce_metadata(obj)
            md.setdefault("row_index", i)

            # Handy defaults for your schema
            md.setdefault("doc_id", obj.get("doc_id"))
            md.setdefault("chunk_id", obj.get("chunk_id"))
            md.setdefault("dish_name", obj.get("dish_name"))
            md.setdefault("section", obj.get("section"))
            md.setdefault("doc_type", obj.get("doc_type"))
            md.setdefault("source", obj.get("source"))

            out.append(Chunk(text=text, metadata=md))

    if not out:
        raise ValueError("No valid chunks found (no non-empty 'text' fields).")

    return out
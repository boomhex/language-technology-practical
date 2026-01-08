from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from meta_filter import MetaFilter


def _extract_json_object(text: str) -> Dict[str, Any]:
    """
    Robustly extracts the first JSON object from a model response.
    Raises ValueError if none found or not parseable.
    """
    print(text)
    #m = _JSON_RE.search(text or "")
    # if not m:
    #     raise ValueError("No JSON object found in LLM output.")
    blob = text.group(0)
    return json.loads(blob)


def _validate_and_normalize(
    raw: Dict[str, Any],
    allowed_sections: Sequence[str],
    allowed_tags: Sequence[str],
) -> Dict[str, Any]:
    """
    Enforces a strict schema and maps values to canonical forms.
    Invalid fields are ignored.
    """
    out: Dict[str, Any] = {}

    # doc_id: optional string
    doc_id = raw.get("doc_id", None)
    if isinstance(doc_id, str) and doc_id.strip():
        out["doc_id"] = doc_id.strip()

    # doc_type: force to "recipe" (or allow null)
    doc_type = raw.get("doc_type", "recipe")
    if doc_type is None:
        out["doc_type"] = "recipe"
    elif isinstance(doc_type, str):
        out["doc_type"] = "recipe" if doc_type.strip() == "" else doc_type.strip()
    else:
        out["doc_type"] = "recipe"

    # source: optional string
    source = raw.get("source", None)
    if isinstance(source, str) and source.strip():
        out["source"] = source.strip()

    # region: optional string
    region = raw.get("region", None)
    if isinstance(region, str) and region.strip():
        out["region"] = region.strip()

    # section: must be in allowed_sections or null
    section = raw.get("section", None)
    if section is None:
        out["section"] = None
    elif isinstance(section, str) and section in allowed_sections:
        out["section"] = section
    else:
        out["section"] = None

    # tags_any / tags_all / tags_not: lists of allowed tag strings
    def norm_tags(key: str) -> Optional[Tuple[str, ...]]:
        val = raw.get(key, None)
        if val is None:
            return None
        if not isinstance(val, list):
            return None
        cleaned = [t for t in val if isinstance(t, str) and t in allowed_tags]
        cleaned = sorted(set(cleaned))
        return tuple(cleaned) if cleaned else None

    out["tags_any"] = norm_tags("tags_any")
    out["tags_all"] = norm_tags("tags_all")
    out["tags_not"] = norm_tags("tags_not")

    return out


def _apply_patch(base: MetaFilter, patch: Dict[str, Any]) -> MetaFilter:
    """
    Applies validated patch values to MetaFilter.
    """
    flt = base

    if patch.get("doc_id") is not None:
        flt = replace(flt, doc_id=patch["doc_id"])

    if patch.get("doc_type") is not None:
        flt = replace(flt, doc_type=patch["doc_type"])

    if patch.get("source") is not None:
        flt = replace(flt, source=patch["source"])

    # region can be null; set only if string given
    if patch.get("region") is not None:
        flt = replace(flt, region=patch["region"])

    # section can be None; set only if not None
    if patch.get("section") is not None:
        flt = replace(flt, section=patch["section"])

    if patch.get("tags_any") is not None:
        flt = replace(flt, tags_any=patch["tags_any"])

    if patch.get("tags_all") is not None:
        flt = replace(flt, tags_all=patch["tags_all"])

    if patch.get("tags_not") is not None:
        flt = replace(flt, tags_not=patch["tags_not"])

    return flt




def classify_to_filter(
    model,
    query: str,
    *,
    allowed_sections: Sequence[str],
    allowed_tags: Sequence[str],
    default_doc_type: str = "recipe",
) -> MetaFilter:
    """
    Always calls the LLM, then validates and returns a MetaFilter.
    """
    prompt = (
        "Convert the query into a metadata filter.\n"
        "Return ONLY one JSON object with keys: section, tags_any, tags_not.\n"
        f"Allowed sections: {allowed_sections}\n"
        f"Allowed tags: {allowed_tags}\n"
        'Example: Query="tiramisu without eggs" -> {"section": null, "tags_any": ["eggless"], "tags_not": []}\n'
        f'Query="{query}"\n'
        "JSON:"
    )

    raw_text = model.call_llm(prompt, )

    raw_obj = _extract_json_object(raw_text)
    patch = _validate_and_normalize(raw_obj, allowed_sections, allowed_tags)

    base = MetaFilter(doc_type=default_doc_type)
    return _apply_patch(base, patch)
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class MetaFilter:
    # Exact matches
    doc_id: Optional[str] = None
    doc_type: Optional[str] = None
    source: Optional[str] = None
    section: Optional[str] = None
    region: Optional[str] = None

    # String contains (case-insensitive)
    dish_name_contains: Optional[str] = None

    # Tag logic
    tags_any: Optional[Sequence[str]] = None     # keep if chunk contains ANY of these tags
    tags_all: Optional[Sequence[str]] = None     # keep if chunk contains ALL of these tags
    tags_not: Optional[Sequence[str]] = None     # drop if chunk contains ANY of these tags


def _norm(s: Optional[str]) -> Optional[str]:
    return s.lower() if isinstance(s, str) else s

def _norm_list(xs: Optional[Sequence[str]]) -> Optional[List[str]]:
    if xs is None:
        return None
    return [str(x).lower() for x in xs]


def match_chunk_meta(md: Dict[str, Any], flt: MetaFilter) -> bool:
    if flt.doc_id is not None and md.get("doc_id") != flt.doc_id:
        return False
    if flt.doc_type is not None and md.get("doc_type") != flt.doc_type:
        return False
    if flt.source is not None and md.get("source") != flt.source:
        return False
    if flt.section is not None and md.get("section") != flt.section:
        return False
    if flt.region is not None and md.get("region") != flt.region:
        return False

    if flt.dish_name_contains is not None:
        name = _norm(md.get("dish_name") or "")
        if _norm(flt.dish_name_contains) not in name:
            return False

    tags = md.get("tags") or []
    tags_set = {str(t).lower() for t in tags}

    any_need = set(_norm_list(flt.tags_any) or [])
    all_need = set(_norm_list(flt.tags_all) or [])
    not_have = set(_norm_list(flt.tags_not) or [])

    if any_need and tags_set.isdisjoint(any_need):
        return False
    if all_need and not all_need.issubset(tags_set):
        return False
    if not_have and not tags_set.isdisjoint(not_have):
        return False

    return True


def filter_ranked(
    ranked_idxs: Sequence[int],
    chunks: Sequence[Any],  # expects chunks[i].metadata dict
    flt: MetaFilter,
    top_k: int,
) -> List[int]:
    """
    Keeps original ranking order, filters by metadata, returns first top_k indices.
    """
    out: List[int] = []
    for idx in ranked_idxs:
        if idx < 0 or idx >= len(chunks):
            continue
        md = getattr(chunks[idx], "metadata", {}) or {}
        if match_chunk_meta(md, flt):
            out.append(idx)
            if len(out) >= top_k:
                break
    return out
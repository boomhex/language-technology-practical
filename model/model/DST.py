# dialogue_state.py
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

@dataclass
class DialogueState:
    # What are we discussing?
    topic: Optional[str] = None              # e.g. "tiramisu"
    selected_doc_id: Optional[str] = None    # e.g. recipe_id, chunk source id, etc.

    # Constraints / slots (customize)
    slots: Dict[str, Any] = field(default_factory=dict)
    # Example slots:
    # slots["diet"] = "vegetarian"
    # slots["avoid"] = ["nuts"]
    # slots["servings"] = 4
    # slots["time_limit_min"] = 30

    # Recency memory (keep short)
    last_user: Optional[str] = None
    last_assistant: Optional[str] = None
    last_hits: List[Dict[str, Any]] = field(default_factory=list)  # retrieval metadata

    # Rolling summary for prompt injection (small!)
    summary: str = ""


def enrich_query(user_message: str, state: DialogueState) -> str:
    msg = user_message.strip()

    pronouns = ("it", "that", "this", "those", "these", "they", "them")
    starts_with_pronoun = msg.lower().split()[:1] and msg.lower().split()[0] in pronouns

    if state.topic and (starts_with_pronoun or "step" in msg.lower() or "that" in msg.lower()):
        return f"{state.topic}. {msg}"

    # Include key constraints as retrieval hints
    hints = []
    if "diet" in state.slots:
        hints.append(f"diet: {state.slots['diet']}")
    if "avoid" in state.slots:
        hints.append("avoid: " + ", ".join(state.slots["avoid"]))

    if hints:
        return msg + " (" + "; ".join(hints) + ")"

    return msg
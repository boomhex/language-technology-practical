# memory.py
from generator import GenerativeQA

class ShortTermMemory:
    """
    Simple rolling buffer of the last N messages in the conversation.
    Each entry: {"role": "user" | "assistant", "text": str}
    """
    def __init__(self, model, max_size: int = 10) -> None:
        self.max_size = max_size
        self.buffer: list[dict[str, str]] = []
        self.model = model

    def add(self, role: str, text: str) -> None:
        self.buffer.append({"role": role, "text": text})
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def get_context(self) -> str:
        """
        Simple textual representation of the conversation so far,
        in case you want to feed it into the model later.
        """
        #return "\n".join(m["text"] for m in self.buffer if m.get("role") == "user")
        hist = "\n".join(f"{m['text']}" for m in self.buffer)
        sum = self.model.summarize(hist)
        return sum
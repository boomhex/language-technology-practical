from dataclasses import dataclass, field
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

class DialogueState:
    dish: Optional[str] = None
    intent: Optional[str] = None
    constraints: Dict[str, str] = []
    history: List[str] = []


class DialogueStateTracking:
    def __init__(self, device: str = "cpu", model_name: str = "google/flan-t5-large"):
        self.state = DialogueState()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

        self.pipe = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if device == "cuda" else -1,
        )

    def _generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        return self.pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            do_sample=False,
        )[0]["generated_text"].strip()

    def update_state(self, user_input: str) -> None:
        self.state.history.append(user_input)

        prompt = (
            "Extract the dish mentioned in the conversation.\n"
            f"Conversation:\n{self.state.history[-1]}\n\n"
            "Return ONLY the dish name or 'none'."
        )
        dish = self._generate(prompt)
        if dish.lower() != "none":
            self.state.dish = dish
        return dish

    def rewrite_for_retrieval(self, user_query: str) -> str:
        # Make the query standalone for RAG retrieval
        dish = self.state.dish or "the discussed dish"
        prompt = (
            "Rewrite the user query into a single, standalone search query.\n"
            f"Dish: {dish}\n"
            f"User query: {user_query}\n\n"
            "Return ONLY the rewritten search query."
        )
        return self._generate(prompt)


# Demo
dst = DialogueStateTracking(device="cpu")
query = input("? ")
while query != '\n':
    print(dst.rewrite_for_retrieval(query))
    print(dst.update_state(query))
    query = input("? ")
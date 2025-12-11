import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

class GenerativeQA:
    def __init__(self, device: str = "cpu", model_name: str = "google/flan-t5-small") -> None:
        device_obj = torch.device(device if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device_obj)

        self.pipe = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if device_obj.type == "cuda" else -1,
        )

    def answer(self, question: str, contexts: list[str], conversation_context="") -> str:
        """
        Answer a question given a list of retrieved context chunks.
        """
        print(conversation_context)
        joined_context = "\n".join(contexts)

        prompt = (
            f"question: {question}\n"
            f"context: {joined_context}\n"
            f"conversation history: {conversation_context}\n\n"
            "Answer the question using only the information provided in the context combined with the history. "
            "If the answer cannot be derived from the context or conversation history, say that the context does not contain the required information. "
            "If the last conversation answer is not devinitive, then try to answer that question again."
            # "Within the context, the ingredients of a recipe may appear inside a sentence such as "
            # "'The ingredients ingredient 1: eggs, ingredient 2: flour, ingredient 3: sugar, ..., ingredient n: ...', "
            # "where each ingredient is written as 'ingredient k: <name>'. "
            # "When the question asks for the ingredients, you must:\n"
            # "1) Find all occurrences of 'ingredient k: <name>' in the context (for any k),\n"
            # "2) Extract only the <name> parts, in order of their numbering, and\n"
            # "3) Answer in the form: 'the ingredients that you need are X, Y and Z', listing only the ingredient names in a natural enumeration.\n"
            # "Do not add ingredients or information that are not explicitly listed in the context. "
            # "Do not infer or assume missing ingredients. "
            # "If no ingredients in the format 'ingredient k: <name>' are present, say that the context does not provide the ingredients. "
            "Follow all these rules precisely."
        )
        out = self.pipe(
            prompt,
            max_new_tokens=250,
            num_beams=4,
            do_sample=False,
        )[0]["generated_text"]
        return out.strip()
    

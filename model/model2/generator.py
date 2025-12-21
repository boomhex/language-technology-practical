import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

class GenerativeQA:
    def __init__(self, device: str = "cpu", model_name: str = "google/flan-t5-large") -> None:
        device_obj = torch.device(device if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device_obj)

        self.pipe = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if device_obj.type == "cuda" else -1,
        )

    def summarize(self, conversation_context: str = "") -> str:
        prompt = (
            "Task: Extract and summarize ONLY the questions asked.\n"
            "Rules:\n"
            "- Do NOT include ingredients, quantities, units, or instructions\n"
            "- Do NOT copy food items\n"
            "- Output ONLY a short question summary\n"
            "- If no question is present, output an ONLY an empty string\n\n"
            "Conversation history:\n"
            f"{conversation_context}\n\n"
            "Output:"
        )

        out = self.pipe(
            prompt,
            max_new_tokens=50,
            num_beams=4,
            do_sample=False,
        )[0]["generated_text"]

        return out.strip()

    def rewrite(self, question: str, conversation_context: str = "") -> str:
        prompt = (
            "You are a query rewriter for a recipe retrieval system.\n\n"
            "Do not change any words in the user question.\n"
            "get the last recipe from the Conversation History.\n"
            "The conversation history consist of user inputs and model answers, use these to get your answer.\n"
            "The user_question is copied without changing\n"
            "If the question has a different recipe than in the history, then use the new recipe from the question\n"
            "If no recipe name is mentioned, output: user_question\n\n"
            "Conversation history:\n"
            f"{conversation_context}\n\n"
            "User_question:\n"
            f"{question}\n\n"
            "output the found recipe and the last user question.\n"
        )

        out = self.pipe(
            prompt,
            max_new_tokens=50,
            num_beams=4,
            do_sample=False,
        )[0]["generated_text"]

        return out.strip()

    def answer(self, question: str, contexts: list[str]) -> str:
        """
        Answer a question given a list of retrieved context chunks.
        """
        joined_context = "\n".join(contexts)

        prompt = (
            f"question: {question}\n"
            f"context: {joined_context}\n\n"
            "Answer the question using only the information provided in the context."
            "If the answer cannot be derived from the context, say that the context does not contain the required information. "
            "Within the context, the ingredients of a recipe may appear inside a sentence such as "
            "'The ingredients ingredient 1: eggs, ingredient 2: flour, ingredient 3: sugar, ..., ingredient n: ...', "
            "where each ingredient is written as 'ingredient k: <name>'. "
            "When the question asks for the ingredients, you must:\n"
            "1) Find all occurrences of 'ingredient k: <name>' in the context (for any k),\n"
            "2) Extract only the <name> parts, in order of their numbering, and\n"
            "3) Answer in the form: 'the ingredients that you need are X, Y and Z', listing only the ingredient names in a natural enumeration.\n"
            "Do not add ingredients or information that are not explicitly listed in the context. "
            "Do not infer or assume missing ingredients. "
            "If no ingredients in the format 'ingredient k: <name>' are present, say that the context does not provide the ingredients. "
            "Follow all these rules precisely."
        )
        out = self.pipe(
            prompt,
            max_new_tokens=250,
            num_beams=4,
            do_sample=False,
        )[0]["generated_text"]
        return out.strip()
    

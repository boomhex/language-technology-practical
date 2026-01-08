import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class Phi3Caller:
    def __init__(self, device: str = "auto") -> None:
        self.model_name = "microsoft/Phi-3-mini-4k-instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map="auto" if device == "auto" else None,
        )
        if device != "auto":
            self.model = self.model.to(device)

    def call_llm(self, prompt: str, *, max_new_tokens: int = 128) -> str:
        messages = [
            {"role": "system", "content": "You are a strict JSON generator. Output only JSON."},
            {"role": "user", "content": prompt},
        ]
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(rendered, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
            )

        # Phi-style models often include the prompt in the decoded text; you can
        # keep your JSON-extractor as a post-step.
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
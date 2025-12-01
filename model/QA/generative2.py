import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline
)
from utils import (
    max_over_ground_truths,
    compute_f1,
    read_file,
    load_questions,
    compute_exact_match
)

class GenerativeModel:
    def __init__(self,
                 device,
                 generative_model_name) -> None:
        device = torch.device(device if torch.cuda.is_available() else "cpu")
                                        # set tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(generative_model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            generative_model_name).to(device)
                                        # Set the pipeline
        self.pipeline = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if device.type == "cuda" else -1)

    def __call__(self, prompt: str) -> str:
        try:
            out = self.pipeline(
                prompt,
                max_new_tokens=128, # max answer length
                num_beams=4, # 4 candidates
                do_sample=False, #deterministic
            )[0]["generated_text"]
            prediction_text = out.strip()
        except Exception as e:
            prediction_text = ""

        return prediction_text

if __name__ == "__main__":
    gen_model = GenerativeModel(
        "cpu",
        "t5-small"
    )
    ans = gen_model("context: My name is Timo. Question: what is my name?")
    print(ans)

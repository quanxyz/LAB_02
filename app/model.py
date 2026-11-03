from pathlib import Path
from typing import Any, Dict
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


class QuestionAnsweringModel:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.model_path = self._load_model_path(config_path)
        print(f"Loading QA model: {self.model_path} ...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_path).to(self.device)
        self.model.eval()
        print(f"Model loaded successfully on {self.device}.")

    def _load_model_path(self, config_path: str) -> str:
        path = Path(config_path)
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                if isinstance(cfg, dict):
                    if "model" in cfg and isinstance(cfg["model"], dict):
                        return cfg["model"].get(
                            "model_path", "distilbert/distilbert-base-cased-distilled-squad"
                        )
                    if "model_path" in cfg:
                        return cfg["model_path"]
            except Exception as e:
                print(f"Warning: Failed to parse config file ({e}). Using default.")
        return "distilbert/distilbert-base-cased-distilled-squad"

    def predict(self, question: str, context: str) -> Dict[str, Any]:
        cleaned_question = question.strip()
        cleaned_context = context.strip()

        inputs = self.tokenizer(
            cleaned_question,
            cleaned_context,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]

        start_idx = int(torch.argmax(start_logits))
        end_idx = int(torch.argmax(end_logits))

        if end_idx < start_idx:
            end_idx = start_idx

        start_prob = torch.softmax(start_logits, dim=-1)[start_idx].item()
        end_prob = torch.softmax(end_logits, dim=-1)[end_idx].item()
        score = round(float(start_prob * end_prob), 4)

        answer_tokens = inputs.input_ids[0, start_idx : end_idx + 1]
        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()

        start_char = cleaned_context.lower().find(answer.lower()) if answer else -1
        end_char = start_char + len(answer) if start_char != -1 else 0
        if start_char == -1:
            start_char = 0

        return {
            "answer": answer,
            "score": score,
            "start": start_char,
            "end": end_char,
        }

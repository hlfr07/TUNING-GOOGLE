from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

MODEL_NAME = "google/gemma-3-270m-it"

dataset = load_dataset(
    "json",
    data_files="all.jsonl"
)["train"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token


def tokenize(example):
    prompt = example["prompt"]
    continuation = example["continuation"]

    full_text = prompt + continuation

    tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=512,
        padding="max_length"
    )

    prompt_tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=512
    )

    labels = tokens["input_ids"].copy()

    prompt_len = len(prompt_tokens["input_ids"])
    labels[:prompt_len] = [-100] * prompt_len

    tokens["labels"] = labels
    return tokens


tokenized = dataset.map(tokenize)

labels = np.array(tokenized[0]["labels"])
trainable_tokens = (labels != -100).sum()

print("🧠 TOKENS ENTRENABLES:", trainable_tokens)

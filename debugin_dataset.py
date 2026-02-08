from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

MODEL_NAME = "google/gemma-3-270m-it"

print("📥 Cargando dataset...")
dataset = load_dataset("json", data_files="all.jsonl")["train"]

print(dataset)
print("\n🔎 Ejemplo crudo:")
print(dataset[0])

# =========================
# TOKENIZER
# =========================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)

tokenizer.pad_token = tokenizer.eos_token

# =========================
# TOKENIZAR
# =========================
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

    # Tokenizar solo prompt
    prompt_tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=512
    )

    labels = tokens["input_ids"].copy()

    prompt_len = len(prompt_tokens["input_ids"])

    # Enmascarar prompt
    labels[:prompt_len] = [-100] * prompt_len

    tokens["labels"] = labels

    return tokens

print("\n⚙️ Tokenizando...")
tokenized = dataset.map(tokenize)

print("\n🔎 Keys:", tokenized[0].keys())

# =========================
# DEBUG LABELS
# =========================
labels = np.array(tokenized[0]["labels"])

trainable_tokens = (labels != -100).sum()

print("\n🧠 TOKENS ENTRENABLES:", trainable_tokens)

if trainable_tokens == 0:
    print("❌ ERROR: No hay tokens entrenables → loss será 0")
else:
    print("✅ Dataset correcto para entrenar")

# =========================
# VER TEXTO DETOKENIZADO
# =========================
decoded = tokenizer.decode(
    tokenized[0]["input_ids"],
    skip_special_tokens=False
)

print("\n🔎 Texto tokenizado:")
print(decoded)

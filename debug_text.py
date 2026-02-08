from datasets import load_dataset

dataset = load_dataset(
    "ArcticHuaji/gemma-3-270m-4b-it-data",
    split="train"
)

print("🔎 PROMPT:")
print(dataset[0]["prompt"])

print("\n🔎 CONTINUATION:")
print(dataset[0]["continuation"])

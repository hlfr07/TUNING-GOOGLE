from datasets import load_dataset

dataset = load_dataset(
    "ArcticHuaji/gemma-3-270m-4b-it-data"
)

# print(dataset)
# print(dataset["train"][0])

print(dataset["train"][0]["prompt"])
print("-----")
print(dataset["train"][0]["continuation"])


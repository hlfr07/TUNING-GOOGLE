from datasets import load_dataset
from transformers import AutoTokenizer

# Cargar tokenizer
tokenizer = AutoTokenizer.from_pretrained("./gemma-3-270m-it", use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Cargar solo un ejemplo del dataset
dataset = load_dataset("json", data_files="all.jsonl")["train"]
example = dataset[0]

print("🔎 PROMPT:")
print(repr(example["prompt"]))
print("\n🔎 CONTINUATION:")  
print(repr(example["continuation"]))

# Función de tokenización corregida
def tokenize(example):
    prompt = str(example["prompt"])
    continuation = str(example["continuation"])
    
    # Tokenizar prompt y continuation por separado (SIN padding)
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    continuation_tokens = tokenizer(continuation, add_special_tokens=False)["input_ids"]
    
    print(f"\n📊 STATS:")
    print(f"   Prompt tokens: {len(prompt_tokens)}")
    print(f"   Continuation tokens: {len(continuation_tokens)}")
    
    # Combinar tokens
    full_tokens = prompt_tokens + continuation_tokens
    
    # Truncar si es necesario
    if len(full_tokens) > 512:
        full_tokens = full_tokens[:512]
    
    # Crear labels: -100 para prompt, tokens reales para continuation
    labels = [-100] * len(prompt_tokens) + continuation_tokens
    if len(labels) > 512:
        labels = labels[:512]
    
    # Hacer padding manualmente al final
    while len(full_tokens) < 512:
        full_tokens.append(tokenizer.pad_token_id)
        labels.append(-100)  # No entrenar en padding
    
    # Crear attention mask
    attention_mask = [1 if token != tokenizer.pad_token_id else 0 for token in full_tokens]
    
    print(f"   Tokens entrenables: {sum(1 for x in labels if x != -100)}")
    print(f"   Total tokens: {len(full_tokens)}")
    print(f"   Attention tokens: {sum(attention_mask)}")
    
    return {
        "input_ids": full_tokens,
        "attention_mask": attention_mask, 
        "labels": labels
    }

# Probar tokenización
result = tokenize(example)

print(f"\n✅ RESULTADO:")
print(f"   input_ids length: {len(result['input_ids'])}")
print(f"   labels length: {len(result['labels'])}")
print(f"   attention_mask length: {len(result['attention_mask'])}")

# Verificar que no hay muchos padding al inicio
first_10_tokens = [tokenizer.decode([token]) for token in result['input_ids'][:10]]
print(f"\n🔎 PRIMEROS 10 TOKENS: {first_10_tokens}")
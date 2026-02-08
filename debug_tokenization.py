#!/usr/bin/env python3
"""
Debug específico de la tokenización
"""
import json
from transformers import AutoTokenizer
from datasets import load_dataset

MODEL_PATH = "./gemma-3-270m-it"
DATA_PATH = "./all.jsonl"

print("🔍 DEBUGGING TOKENIZACIÓN ESPECÍFICO")
print("="*50)

# Cargar tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
print(f"Vocab size: {tokenizer.vocab_size}")
print(f"Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

# Cargar dataset
dataset = load_dataset("json", data_files=DATA_PATH)["train"]
print(f"Dataset loaded: {len(dataset)} examples")

# Tomar primer ejemplo
example = dataset[0]
prompt = str(example["prompt"]).strip()
continuation = str(example["continuation"]).strip()

print(f"\n📝 EJEMPLO ORIGINAL:")
print(f"Prompt ({len(prompt)} chars):")
print(repr(prompt))
print(f"\nContinuation ({len(continuation)} chars):")
print(repr(continuation[:200]))

# Tokenizar paso a paso
print(f"\n🔢 TOKENIZACIÓN PASO A PASO:")

# 1. Tokenizar prompt
prompt_tokens = tokenizer(prompt, add_special_tokens=False, truncation=False)["input_ids"]
print(f"Prompt tokens: {len(prompt_tokens)} tokens")
print(f"Primeros 10: {prompt_tokens[:10]}")

# 2. Tokenizar continuation  
continuation_tokens = tokenizer(continuation, add_special_tokens=False, truncation=False)["input_ids"]
print(f"Continuation tokens: {len(continuation_tokens)} tokens")
print(f"Primeros 10: {continuation_tokens[:10]}")

# 3. Verificar si hay tokens válidos
print(f"\n✅ VERIFICACIONES:")
print(f"Prompt vacío: {len(prompt_tokens) == 0}")
print(f"Continuation vacía: {len(continuation_tokens) == 0}")

# 4. Simular la lógica completa de la función tokenize()
print(f"\n🔄 SIMULANDO FUNCIÓN TOKENIZE():")

# Límites
max_prompt_length = 450
if len(prompt_tokens) > max_prompt_length:
    prompt_tokens = prompt_tokens[:max_prompt_length]
    print(f"Prompt truncado a: {len(prompt_tokens)} tokens")

# Combinar
full_tokens = prompt_tokens + continuation_tokens
print(f"Tokens combinados: {len(full_tokens)} tokens")

# Truncar si necesario
if len(full_tokens) > 512:
    max_total_prompt = min(len(prompt_tokens), 512 - 20)
    prompt_tokens = prompt_tokens[:max_total_prompt]
    continuation_tokens = continuation_tokens[:512 - len(prompt_tokens)]
    full_tokens = prompt_tokens + continuation_tokens
    print(f"Después de truncar: prompt={len(prompt_tokens)}, continuation={len(continuation_tokens)}, total={len(full_tokens)}")

# Crear labels
labels = [-100] * len(prompt_tokens) + continuation_tokens
print(f"Labels: {len(labels)} elementos")

# Padding
while len(full_tokens) < 512:
    full_tokens.append(tokenizer.pad_token_id)
    labels.append(-100)

print(f"Después de padding: {len(full_tokens)} tokens, {len(labels)} labels")

# ANÁLISIS FINAL
trainable_tokens = sum(1 for label in labels if label != -100)
print(f"\n🎯 RESULTADO FINAL:")
print(f"Tokens entrenables: {trainable_tokens}")
print(f"Tokens no entrenables (padding + prompt): {512 - trainable_tokens}")
print(f"Porcentaje entrenable: {trainable_tokens/512*100:.1f}%")

if trainable_tokens == 0:
    print(f"❌ PROBLEMA: No hay tokens entrenables!")
    print(f"   Esto explica loss = 0.0")
else:
    print(f"✅ Hay tokens entrenables, el problema debe ser otro")

# Verificar si los tokens de continuation son válidos  
continuation_labels = [label for label in labels if label != -100]
print(f"\nTokens de continuation para entrenar: {continuation_labels[:10]}...")

# Verificar que estos tokens estén en el rango válido del vocabulario
invalid_tokens = [token for token in continuation_labels if token >= tokenizer.vocab_size or token < 0]
if invalid_tokens:
    print(f"⚠️ Tokens inválidos encontrados: {len(invalid_tokens)}")
    print(f"Ejemplos: {invalid_tokens[:10]}")
else:
    print(f"✅ Todos los tokens están en el rango válido del vocabulario")

print("\n" + "="*50)
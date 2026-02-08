#!/usr/bin/env python3
"""
Script para debuggear el dataset y encontrar por qué loss = 0
"""
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import json

# Configuración
MODEL_PATH = "./gemma-3-270m-it"
DATA_PATH = "./all.jsonl"

# Cargar tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("🔍 DEBUGGING DATASET")
print("="*50)

# 1. Verificar estructura del dataset original
print("\n1. VERIFICANDO ESTRUCTURA ORIGINAL:")
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    line = f.readline()
    sample = json.loads(line)
    print(f"Keys del JSONL: {list(sample.keys())}")
    print(f"Prompt ejemplo: {sample['prompt'][:100]}...")
    print(f"Continuation ejemplo: {sample['continuation'][:100]}...")

# 2. Cargar dataset
dataset = load_dataset("json", data_files=DATA_PATH)["train"]
print(f"\n2. Dataset cargado: {len(dataset)} ejemplos")

# 3. Función de tokenización ORIGINAL (copiada del script)
def tokenize(example):
    prompt = str(example["prompt"]).strip()
    continuation = str(example["continuation"]).strip()
    
    # Verificar que no estén vacíos
    if not prompt or not continuation:
        return {
            "input_ids": [tokenizer.pad_token_id] * 512,
            "attention_mask": [0] * 512,
            "labels": [-100] * 512
        }
    
    # Tokenizar
    prompt_tokens = tokenizer(prompt, add_special_tokens=False, truncation=False)["input_ids"]
    continuation_tokens = tokenizer(continuation, add_special_tokens=False, truncation=False)["input_ids"]
    
    if len(continuation_tokens) == 0:
        return {
            "input_ids": [tokenizer.pad_token_id] * 512,
            "attention_mask": [0] * 512,
            "labels": [-100] * 512
        }
    
    # Limites
    max_prompt_length = 450
    if len(prompt_tokens) > max_prompt_length:
        prompt_tokens = prompt_tokens[:max_prompt_length]
    
    # Combinar
    full_tokens = prompt_tokens + continuation_tokens
    
    # Truncar
    if len(full_tokens) > 512:
        max_total_prompt = min(len(prompt_tokens), 512 - 20)
        prompt_tokens = prompt_tokens[:max_total_prompt]
        continuation_tokens = continuation_tokens[:512 - len(prompt_tokens)]
        full_tokens = prompt_tokens + continuation_tokens
    
    # Labels: -100 para prompt, tokens reales para continuation
    labels = [-100] * len(prompt_tokens) + continuation_tokens
    
    # Padding
    while len(full_tokens) < 512:
        full_tokens.append(tokenizer.pad_token_id)
        labels.append(-100)
    
    # Attention mask
    attention_mask = [1 if token != tokenizer.pad_token_id else 0 for token in full_tokens]
    
    return {
        "input_ids": full_tokens,
        "attention_mask": attention_mask, 
        "labels": labels
    }

# 4. Analizar algunos ejemplos ANTES de tokenizar
print("\n3. ANALIZANDO EJEMPLOS RAW:")
for i in range(3):
    example = dataset[i]
    prompt = str(example["prompt"]).strip()
    continuation = str(example["continuation"]).strip()
    
    print(f"\nEjemplo {i+1}:")
    print(f"  Prompt length: {len(prompt)} chars")
    print(f"  Continuation length: {len(continuation)} chars")
    print(f"  Prompt: {prompt[:100]}...")
    print(f"  Continuation: {continuation[:100]}...")

# 5. Tokenizar y analizar
print("\n4. ANALIZANDO EJEMPLOS TOKENIZADOS:")
for i in range(3):
    example = dataset[i]
    tokenized = tokenize(example)
    
    input_ids = tokenized["input_ids"]
    labels = tokenized["labels"]
    attention_mask = tokenized["attention_mask"]
    
    # Contar estadísticas
    total_tokens = len(input_ids)
    useful_tokens = sum(attention_mask)
    trainable_tokens = sum(1 for label in labels if label != -100)
    padding_tokens = total_tokens - useful_tokens
    
    print(f"\nEjemplo {i+1} tokenizado:")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Tokens útiles: {useful_tokens}")
    print(f"  Tokens entrenables: {trainable_tokens}")
    print(f"  Tokens padding: {padding_tokens}")
    print(f"  % entrenable: {trainable_tokens/total_tokens*100:.1f}%")
    
    # Mostrar algunos tokens y labels
    print(f"  Primeros 20 tokens: {input_ids[:20]}")
    print(f"  Primeros 20 labels: {labels[:20]}")
    
    # Verificar si hay ALGÚN token entrenable
    if trainable_tokens == 0:
        print(f"  ⚠️ PROBLEMA: NO HAY TOKENS ENTRENABLES")
    
    # Decodificar para ver el texto
    text = tokenizer.decode(input_ids, skip_special_tokens=True)
    print(f"  Texto: {text[:100]}...")

# 6. Estadísticas globales
print("\n5. APLICANDO TOKENIZACIÓN A TODO EL DATASET:")
sample_size = 100  # Solo una muestra para no tardar mucho
sample_dataset = dataset.select(range(min(sample_size, len(dataset))))

tokenized_sample = sample_dataset.map(tokenize, remove_columns=sample_dataset.column_names)

# Contar estadísticas generales
total_trainable = 0
total_examples = 0
problematic_examples = 0

for example in tokenized_sample:
    labels = example["labels"]
    trainable_in_example = sum(1 for label in labels if label != -100)
    
    total_trainable += trainable_in_example
    total_examples += 1
    
    if trainable_in_example == 0:
        problematic_examples += 1

print(f"\nESTADÍSTICAS DE {sample_size} EJEMPLOS:")
print(f"  Ejemplos problemáticos (sin tokens entrenables): {problematic_examples}")
print(f"  Ejemplos válidos: {total_examples - problematic_examples}")
print(f"  Promedio tokens entrenables por ejemplo: {total_trainable/total_examples:.1f}")
print(f"  Porcentaje problemático: {problematic_examples/total_examples*100:.1f}%")

if problematic_examples == total_examples:
    print("\n💥 PROBLEMA ENCONTRADO:")
    print("   TODOS los ejemplos tienen 0 tokens entrenables!")
    print("   Esto explica por qué loss = 0.0")
elif problematic_examples > total_examples * 0.5:
    print(f"\n⚠️ MUCHOS ejemplos problemáticos ({problematic_examples/total_examples*100:.1f}%)")
else:
    print(f"\n✅ La mayoría de ejemplos parecen válidos")

print("\n" + "="*50)
print("DEBUG COMPLETADO")
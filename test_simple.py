#!/usr/bin/env python3
"""
TEST ULTRA SIMPLIFICADO - Minimal setup para identificar el problema
"""
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
from datasets import load_dataset  
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

print("🧪 ENTRENAMIENTO MINIMAL PARA TESTING")
print("="*50)

# Configuración básica
MODEL_PATH = "./gemma-3-270m-it"
DATA_PATH = "./all.jsonl" 
OUTPUT_DIR = "./test_output"

# Cargar tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Cargar modelo - FLOAT32 para máxima estabilidad
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float32,  # Forzar float32
    device_map="auto",
    attn_implementation="eager"
)

print(f"✅ Modelo cargado en {model.device}")

# Configuración mínima
model.config.use_cache = False
model.train()

# Dataset - solo 100 ejemplos para testing rápido
dataset = load_dataset("json", data_files=DATA_PATH)["train"]
dataset = dataset.select(range(100))  # Solo 100 ejemplos
print(f"📊 Dataset reducido: {len(dataset)} ejemplos")

# Función de tokenización ULTRA SIMPLE
def simple_tokenize(example):
    prompt = str(example["prompt"]).strip()
    continuation = str(example["continuation"]).strip()
    
    # Concatenar directamente  
    text = prompt + continuation
    
    # Tokenizar todo junto con límite estricto
    tokens = tokenizer(
        text,
        truncation=True,
        max_length=256,  # Más corto para testing
        padding="max_length",
        return_tensors=None
    )
    
    # Labels = input_ids (autoregressive)
    labels = tokens["input_ids"].copy()
    
    # Máscarar prompt (opcional, por ahora entrenar en todo)
    # labels[:len(prompt_tokens)] = [-100] * len(prompt_tokens)
    
    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"], 
        "labels": labels
    }

# Aplicar tokenización
tokenized = dataset.map(simple_tokenize, remove_columns=dataset.column_names)
print(f"🔎 Primer ejemplo tokenizado:")
first = tokenized[0]
trainable = sum(1 for x in first["labels"] if x != -100)
print(f"  Tokens entrenables: {trainable}/{len(first['labels'])}")

if trainable == 0:
    print("❌ PROBLEMA: Aún no hay tokens entrenables!")
    exit(1)

# LoRA super simple  
lora_config = LoraConfig(
    r=4,  # Más pequeño
    lora_alpha=8,
    target_modules=["q_proj"],  # Solo un módulo 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
print(f"🔗 LoRA aplicado: {model.get_nb_trainable_parameters()} params entrenables")

# Training args ULTRA conservadores
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,  # Sin acumulación para simplificar
    num_train_epochs=0.1,  # Solo una fracción de época
    max_steps=10,  # Solo 10 pasos para testing
    logging_steps=1,  # Log cada step
    save_steps=1000,  # No guardar
    fp16=False,
    report_to="none",
    learning_rate=1e-4,  # LR más alto para ver cambios rápido
    max_grad_norm=0.5, 
    warmup_steps=0,  # Sin warmup
    weight_decay=0.0,  # Sin regularización
    dataloader_drop_last=True,
    optim="adamw_torch",
    remove_unused_columns=False
)

# Data collator simple
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Trainer básico
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator
)

# TESTING MANUAL antes de entrenar
print(f"\n🔬 TEST MANUAL:")
batch = tokenized[0]
inputs = {
    "input_ids": torch.tensor([batch["input_ids"]]),
    "attention_mask": torch.tensor([batch["attention_mask"]]),
    "labels": torch.tensor([batch["labels"]])
}

# Mover a device
for key in inputs:
    inputs[key] = inputs[key].to(model.device)

print(f"Input shapes: {[(k, v.shape) for k, v in inputs.items()]}")

# Forward pass manual
with torch.no_grad():
    outputs = model(**inputs)
    print(f"Loss manual: {outputs.loss}")
    print(f"Logits shape: {outputs.logits.shape}")
    print(f"Logits have NaN: {torch.isnan(outputs.logits).any()}")
    print(f"Loss have NaN: {torch.isnan(outputs.loss).any() if outputs.loss is not None else 'No loss'}")

if outputs.loss is None or outputs.loss == 0.0:
    print("❌ PROBLEMA CONFIRMADO: Loss manual también es 0!")
    
    # Debug más profundo
    labels = inputs["labels"]
    active_labels = labels[labels != -100]
    print(f"Labels activos: {len(active_labels)} de {labels.numel()}")
    print(f"Range de labels: min={active_labels.min()}, max={active_labels.max()}")
else:
    print(f"✅ Loss manual parece válido: {outputs.loss}")

print(f"\n🚀 INICIANDO ENTRENAMIENTO TESTING...")

try:
    trainer.train()
    print("✅ Test completado!")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("TEST COMPLETADO")
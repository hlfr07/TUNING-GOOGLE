import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Deshabilitar tensorflow

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
import torch

print("🚀 Iniciando entrenamiento...")

# ------------------------
# RUTAS LOCALES
# ------------------------
MODEL_PATH = "./gemma-3-270m-it"
DATA_PATH = "./all.jsonl"
OUTPUT_DIR = "./outputs"

use_cuda = torch.cuda.is_available()
print(f"📱 CUDA disponible: {use_cuda}")

# ------------------------
# TOKENIZER  
# ------------------------
print("🔧 Cargando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=False
)

# Configurar tokens especiales
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if tokenizer.bos_token is None:
    tokenizer.bos_token = tokenizer.eos_token

print(f"📝 Pad token: {tokenizer.pad_token}")

# ------------------------
# MODELO
# ------------------------
print("🤖 Cargando modelo...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if use_cuda else torch.float32,
    device_map="auto",
    attn_implementation="eager"
)

model.config.use_cache = False
model.gradient_checkpointing_enable()

# ------------------------
# DATASET LOCAL (SOLO PRIMEROS 100)
# ------------------------
print("📊 Cargando dataset...")
dataset = load_dataset(
    "json",
    data_files=DATA_PATH
)["train"]

# Solo tomar los primeros 5000 ejemplos para testing más realista
dataset = dataset.select(range(5000))
print("📊 Filas dataset:", len(dataset))

# ------------------------
# TOKENIZACIÓN CORREGIDA
# ------------------------
def tokenize(example):
    prompt = str(example["prompt"])
    continuation = str(example["continuation"])
    
    # Tokenizar prompt y continuation por separado (SIN padding)
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    continuation_tokens = tokenizer(continuation, add_special_tokens=False)["input_ids"]
    
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
    
    return {
        "input_ids": full_tokens,
        "attention_mask": attention_mask, 
        "labels": labels
    }

print("🔄 Tokenizando dataset...")
dataset = dataset.map(
    tokenize,
    remove_columns=dataset.column_names,
    batched=False
)

print("🔎 Keys:", dataset[0].keys())

# Verificar que tenemos tokens entrenables
trainable_tokens = sum(1 for x in dataset[0]["labels"] if x != -100)
print(f"🧠 Tokens entrenables en primer ejemplo: {trainable_tokens}")

# ------------------------
# CONFIG LORA
# ------------------------
print("⚙️ Configurando LoRA...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ------------------------
# TRAINING ARGUMENTS MUY CONSERVADORES
# ------------------------
print("🎯 Configurando entrenamiento...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,      
    gradient_accumulation_steps=2,      # Más conservador
    num_train_epochs=1,
    max_steps=100,                      # Más pasos para ver progreso real
    logging_steps=10,                   # Log cada 10 pasos
    save_steps=100,                     # No guardar frecuentemente
    fp16=use_cuda,
    learning_rate=1e-5,                 # Learning rate más bajo
    warmup_steps=5,                     # Menos warmup
    report_to="none",
    dataloader_num_workers=0,           # Sin threading
    remove_unused_columns=False         # Mantener columnas
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

# ------------------------
# TRAIN - SOLO 20 PASOS
# ------------------------
print("🔥 ¡INICIANDO ENTRENAMIENTO!")
print("=" * 50)

try:
    trainer.train()
    print("✅ Entrenamiento completado exitosamente!")
except Exception as e:
    print(f"❌ Error durante entrenamiento: {e}")
    import traceback
    traceback.print_exc()

print("🏁 Fin del script")
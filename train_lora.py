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

# ------------------------
# TOKENIZER
# ------------------------
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=False
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ------------------------
# MODELO
# ------------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float32,  # CAMBIO CLAVE: float32 para estabilidad
    device_map="auto",
    attn_implementation="eager"
)

model.config.use_cache = False
model.gradient_checkpointing_enable()

# Asegurar que el modelo esté en modo de entrenamiento
model.train()

# ------------------------
# DATASET LOCAL
# ------------------------
dataset = load_dataset(
    "json",
    data_files=DATA_PATH
)["train"]

print("📊 Filas dataset:", len(dataset))

# ------------------------
# TOKENIZACIÓN CON MASK DE PROMPT CORREGIDA
# ------------------------
def tokenize(example):
    prompt = str(example["prompt"]).strip()
    continuation = str(example["continuation"]).strip()
    
    # Verificar que no estén vacíos
    if not prompt or not continuation:
        # Retornar un ejemplo válido pero que será filtrado
        return {
            "input_ids": [tokenizer.pad_token_id] * 512,
            "attention_mask": [0] * 512,
            "labels": [-100] * 512
        }
    
    # Tokenizar prompt y continuation por separado (SIN padding)
    prompt_tokens = tokenizer(prompt, add_special_tokens=False, truncation=False)["input_ids"]
    continuation_tokens = tokenizer(continuation, add_special_tokens=False, truncation=False)["input_ids"]
    
    # Verificar que continuation no esté vacía
    if len(continuation_tokens) == 0:
        return {
            "input_ids": [tokenizer.pad_token_id] * 512,
            "attention_mask": [0] * 512,
            "labels": [-100] * 512
        }
    
    # Asegurar que haya espacio para al menos algunos tokens de continuation
    max_prompt_length = 450  # Dejar espacio para continuation
    if len(prompt_tokens) > max_prompt_length:
        prompt_tokens = prompt_tokens[:max_prompt_length]
    
    # Combinar tokens
    full_tokens = prompt_tokens + continuation_tokens
    
    # Truncar si es necesario, manteniendo al menos algunos tokens de continuation
    if len(full_tokens) > 512:
        # Asegurar que al menos 20 tokens sean de continuation
        max_total_prompt = min(len(prompt_tokens), 512 - 20)
        prompt_tokens = prompt_tokens[:max_total_prompt]
        continuation_tokens = continuation_tokens[:512 - len(prompt_tokens)]
        full_tokens = prompt_tokens + continuation_tokens
    
    # Crear labels: -100 para prompt, tokens reales para continuation
    labels = [-100] * len(prompt_tokens) + continuation_tokens
    
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

dataset = dataset.map(
    tokenize,
    remove_columns=dataset.column_names
)

# Mezclar el dataset
dataset = dataset.shuffle(seed=42)

print("🔎 Keys:", dataset[0].keys())

# ------------------------
# VERIFICACIÓN DEL DATASET
# ------------------------
def verify_dataset_sample(dataset, num_samples=3):
    print("\n📋 Verificando muestras del dataset...")
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        input_ids = sample["input_ids"]
        labels = sample["labels"]
        attention_mask = sample["attention_mask"]
        
        # Contar tokens útiles vs padding
        useful_tokens = sum(1 for x in attention_mask if x == 1)
        label_tokens = sum(1 for x in labels if x != -100)
        
        print(f"Muestra {i+1}:")
        print(f"  - Tokens útiles: {useful_tokens}/512")
        print(f"  - Tokens a entrenar: {label_tokens}")
        print(f"  - Texto (primeros 100 chars): {tokenizer.decode(input_ids[:20], skip_special_tokens=True)[:100]}...")
        
        # Verificar que no hay valores problemáticos
        if any(x != x for x in input_ids):  # Check for NaN
            print(f"  ⚠️ NaN detectado en input_ids muestra {i+1}")
        if any(x != x for x in labels if x != -100):  # Check for NaN in labels
            print(f"  ⚠️ NaN detectado en labels muestra {i+1}")
            
verify_dataset_sample(dataset)

# ------------------------
# CONFIG LORA
# ------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,  # Dropout moderado
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ------------------------
# TRAINING
# ------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,      # 6GB VRAM ok
    gradient_accumulation_steps=4,      # simula batch grande
    num_train_epochs=1,
    logging_steps=10,
    save_steps=500,
    fp16=False,                         # DESHABILITADO por problemas NaN
    report_to="none",
    # CORRECCIONES IMPORTANTES:
    learning_rate=1e-4,                 # Learning rate funcional (no micro)
    max_grad_norm=1.0,                  # Gradient clipping estándar
    warmup_steps=0,                     # Sin warmup problemático
    weight_decay=0.01,                  # Regularización moderada
    dataloader_drop_last=True,          # Evitar batches irregulares
    save_total_limit=2,                 # Limitar checkpoints
    load_best_model_at_end=False,       # No cargar mejor modelo al final
    optim="adamw_torch",                # Optimizador estable
    remove_unused_columns=False         # Mantener todas las columnas
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# ------------------------
# TRAINER SIMPLIFICADO 
# ------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

# ------------------------
# TRAIN CON MANEJO DE ERRORES
# ------------------------
print("\n🚀 Iniciando entrenamiento...")
print(f"📊 Total ejemplos: {len(dataset)}")
print(f"💾 Pasos totales: {len(dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)}")

try:
    trainer.train()
    print("✅ Entrenamiento completado exitosamente!")
    
except Exception as e:
    print(f"❌ Error durante el entrenamiento: {e}")
    print("💡 Posibles soluciones:")
    print("  1. Reducir batch_size o aumentar gradient_accumulation_steps")
    print("  2. Usar fp16=False si hay problemas de precisión")
    print("  3. Verificar que el dataset no tiene ejemplos corruptos")
    print("  4. Reducir learning_rate aún más")
    
    # Intentar guardar el estado actual si es posible
    try:
        model.save_pretrained(f"{OUTPUT_DIR}/checkpoint-error")
        print(f"💾 Estado guardado en {OUTPUT_DIR}/checkpoint-error")
    except:
        print("❌ No se pudo guardar el checkpoint de emergencia")
    
    raise

# ------------------------
# GUARDAR LoRA + TOKENIZER
# ------------------------
try:
    print("\n💾 Guardando modelo...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Verificar que los archivos se guardaron
    import os
    saved_files = os.listdir(OUTPUT_DIR)
    print(f"✅ LoRA entrenado guardado en {OUTPUT_DIR}")
    print(f"📁 Archivos guardados: {len(saved_files)}")
    
    # Mostrar algunos archivos importantes
    important_files = ['adapter_model.safetensors', 'adapter_config.json', 'tokenizer_config.json']
    for file in important_files:
        if file in saved_files:
            print(f"  ✓ {file}")
        else:
            print(f"  ⚠️ {file} no encontrado")
            
except Exception as e:
    print(f"❌ Error al guardar: {e}")
    
print("\n🎉 Proceso completado!")

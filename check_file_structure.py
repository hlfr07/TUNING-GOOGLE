#!/usr/bin/env python3
"""
Script para verificar la estructura REAL del archivo
"""
import json

print("🔍 INVESTIGANDO ESTRUCTURA REAL DEL ARCHIVO")
print("="*60)

# 1. Verificar si es JSON array o JSONL
try:
    with open('all.jsonl', 'r', encoding='utf-8') as f:
        # Leer todo como JSON array
        data = json.load(f)
    print(f"✅ Archivo es JSON array con {len(data)} elementos")
    
    # Mostrar estructura del primer elemento
    first_item = data[0]
    print(f"\nKeys del primer elemento: {list(first_item.keys())}")
    
    for key, value in first_item.items():
        print(f"\n{key}:")
        print(f"  Tipo: {type(value)}")
        print(f"  Contenido: {str(value)[:200]}...")
        if isinstance(value, str) and len(value) > 200:
            print(f"  ... (total {len(value)} chars)")
    
    # Verificar si tiene la estructura esperada
    if 'prompt' in first_item and 'continuation' in first_item:
        print(f"\n✅ Tiene estructura prompt + continuation")
    elif 'prompt' in first_item:
        print(f"\n⚠️ Solo tiene 'prompt', falta 'continuation'")
    else:
        print(f"\n❌ Estructura inesperada")
        
    # Analizar varios ejemplos
    print(f"\n📊 ANÁLISIS DE PRIMEROS 5 ELEMENTOS:")
    for i in range(min(5, len(data))):
        item = data[i]
        prompt_len = len(item.get('prompt', ''))
        continuation_len = len(item.get('continuation', ''))
        print(f"Elemento {i+1}: prompt={prompt_len} chars, continuation={continuation_len} chars")
        
        # Si continuation está vacía o es muy corta
        if 'continuation' not in item:
            print(f"  ⚠️ Sin campo 'continuation'")
        elif len(item.get('continuation', '')) == 0:
            print(f"  ⚠️ Continuation vacía")
        elif len(item.get('continuation', '')) < 10:
            print(f"  ⚠️ Continuation muy corta: '{item['continuation']}'")

except json.JSONDecodeError as e:
    print(f"❌ Error al leer JSON: {e}")
    # Intentar leer como JSONL línea por línea
    print("Intentando leer como JSONL...")
    try:
        with open('all.jsonl', 'r', encoding='utf-8') as f:
            lines = f.readlines()[:5]  # Solo primeras 5 líneas
            for i, line in enumerate(lines):
                try:
                    item = json.loads(line.strip())
                    print(f"Línea {i+1}: {list(item.keys())}")
                except:
                    print(f"Línea {i+1}: ERROR - no es JSON válido")
    except Exception as e2:
        print(f"❌ Error al leer archivo: {e2}")
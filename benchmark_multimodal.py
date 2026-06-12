import sys
import os
import time
import torch
import gc
import numpy as np
import pandas as pd
import requests
from datasets import load_dataset, Audio
from transformers import WavLMModel, WhisperModel, WhisperProcessor

# ── CONFIGURATION ─────────────────────────────────────────────
OLLAMA_CHAT_URL  = "http://localhost:11434/api/chat"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
# Only running the winning model for the 10k scale-up to save hours of processing time
MODELS           = ["llama3.1:8b"] 
EMBED_MODEL      = "nomic-embed-text"
OUTPUT_DIR       = "extracted_features"
SPLITS           = ["train", "test"] # The dataset has 'train' and 'test' splits

os.makedirs(OUTPUT_DIR, exist_ok=True)

def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(1)

def get_llm_reasoning(text, model_name):
    """Asks the LLM to reason about the emotion in text."""
    prompt = (
        f"Text: \"{text}\"\n"
        "Describe the likely emotional state of the speaker in 3-5 keywords. "
        "Output ONLY the keywords separated by commas. No intro."
    )
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.3}
    }
    try:
        res = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=60)
        return res.json()["message"]["content"].strip()
    except Exception as e:
        return "neutral, calm"

def get_semantic_vector(reasoning_text):
    """Embeds the reasoning text using a reliable embedding model."""
    try:
        res = requests.post(OLLAMA_EMBED_URL, json={"model": EMBED_MODEL, "input": reasoning_text}, timeout=30)
        return np.array(res.json()["embeddings"][0], dtype=np.float32)
    except:
        return np.zeros(768, dtype=np.float32)

print("="*60)
print(" MULTIMODAL FEATURE EXTRACTOR (FULL SCALE)")
print(" Audio -> [WavLM+Wh] + [LLM Reasoning -> Vector]")
print("="*60)

for split_name in SPLITS:
    print(f"\n" + "="*40)
    print(f" PROCESSING SPLIT: {split_name.upper()}")
    print("="*40)
    
    # STAGE 1
    print("\n>>> STAGE 1: Acoustic Extraction...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(device)
    whisper_model = WhisperModel.from_pretrained("openai/whisper-base").to(device)
    whisper_proc  = WhisperProcessor.from_pretrained("openai/whisper-base")

    dataset = load_dataset("stapesai/ssi-speech-emotion-recognition", split=split_name, streaming=True)
    dataset = dataset.cast_column("file_path", Audio(sampling_rate=16000))

    base_data = []
    count = 0
    with torch.no_grad():
        for item in dataset:
            print(f"  [{split_name.upper()}] Processing Audio {count+1}: {item['emotion']}")
            arr = np.array(item['file_path']['array'], dtype=np.float32)
            ten = torch.tensor(arr).unsqueeze(0).to(device)
            
            # Features
            w_out = wavlm(ten).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            f_in  = whisper_proc(arr, sampling_rate=16000, return_tensors="pt").input_features.to(device)
            s_tok = torch.tensor([[1, 1]]).to(device) * whisper_model.config.decoder_start_token_id
            wh_out= whisper_model(f_in, decoder_input_ids=s_tok).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            
            base_data.append({
                "label": item['emotion'], "text": item['text'],
                "wavlm_vector": w_out, "whisper_vector": wh_out
            })
            count += 1

    print("[!] Unloading acoustic models...")
    del wavlm, whisper_model
    clear_gpu()

    # STAGE 2
    print("\n>>> STAGE 2: LLM Reasoning Extraction...")
    for model in MODELS:
        print(f"\nProcessing LLM: {model} for split {split_name}")
        final_recs = []
        for entry in base_data:
            reasoning = get_llm_reasoning(entry['text'], model)
            llm_vec   = get_semantic_vector(reasoning)
            
            rec = entry.copy()
            rec['llm_vector'] = llm_vec
            final_recs.append(rec)
            
        path = os.path.join(OUTPUT_DIR, f"features_{model.replace(':', '_')}_{split_name}.pt")
        torch.save(final_recs, path)
        print(f"  [OK] Saved -> {path}")

print("\n" + "="*60 + "\nCOMPLETE. Now run: python train_fusion.py")


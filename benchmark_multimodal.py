import sys
import os
import time
import torch
import torchaudio
import numpy as np
import pandas as pd
import requests
from datasets import load_dataset
from transformers import WavLMModel, WhisperModel, WhisperProcessor

# ── CONFIGURATION ─────────────────────────────────────────────
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
# Only running the winning model for the 10k scale-up to save hours of processing time
MODELS = ["llama3.1:8b"]
EMBED_MODEL = "nomic-embed-text"
OUTPUT_DIR = "extracted_features"
SPLITS = ["train", "test"]  # The dataset has 'train' and 'test' splits

os.makedirs(OUTPUT_DIR, exist_ok=True)


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(1)


def get_llm_reasoning(text, model_name):
    """Asks the LLM to reason about the emotion in text."""
    prompt = (
        f'Text: "{text}"\n'
        "Describe the likely emotional state of the speaker in 3-5 keywords. "
        "Output ONLY the keywords separated by commas. No intro."
    )
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.3},
    }
    try:
        res = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=60)
        return res.json()["message"]["content"].strip()
    except Exception as e:
        return "neutral, calm"


def get_semantic_vector(reasoning_text):
    """Embeds the reasoning text using a reliable embedding model."""
    try:
        res = requests.post(
            OLLAMA_EMBED_URL,
            json={"model": EMBED_MODEL, "input": reasoning_text},
            timeout=30,
        )
        return np.array(res.json()["embeddings"][0], dtype=np.float32)
    except:
        return np.zeros(768, dtype=np.float32)


print("=" * 60)
print(" MULTIMODAL FEATURE EXTRACTOR (FULL SCALE)")
print(" Audio -> [WavLM+Wh] + [LLM Reasoning -> Vector]")
print("=" * 60)

for split_name in SPLITS:
    print(f"\n" + "=" * 40)
    print(f" PROCESSING SPLIT: {split_name.upper()}")
    print("=" * 40)

    # STAGE 1
    print("\n>>> STAGE 1: Acoustic Extraction...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(device)
    whisper_model = WhisperModel.from_pretrained("openai/whisper-base").to(device)
    whisper_proc = WhisperProcessor.from_pretrained("openai/whisper-base")

    dataset = load_dataset(
        "stapesai/ssi-speech-emotion-recognition", split=split_name, streaming=True
    )
    dataset = dataset.cast_column("file_path", Audio(sampling_rate=16000))

    base_data = []
    count = 0
    with torch.no_grad():
        for item in dataset:
            print(
                f"  [{split_name.upper()}] Processing Audio {count + 1}: {item['emotion']}"
            )
            arr = np.array(item["file_path"]["array"], dtype=np.float32)
            ten = torch.tensor(arr).unsqueeze(0).to(device)

            # Features
            w_out = wavlm(ten).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            f_in = whisper_proc(
                arr, sampling_rate=16000, return_tensors="pt"
            ).input_features.to(device)
            s_tok = (
                torch.tensor([[1, 1]]).to(device)
                * whisper_model.config.decoder_start_token_id
            )
            wh_out = (
                whisper_model(f_in, decoder_input_ids=s_tok)
                .last_hidden_state.mean(dim=1)
                .squeeze()
                .cpu()
                .numpy()
            )

            base_data.append(
                {
                    "label": item["emotion"],
                    "text": item["text"],
                    "wavlm_vector": w_out,
                    "whisper_vector": wh_out,
                }
            )
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
            reasoning = get_llm_reasoning(entry["text"], model)
            llm_vec = get_semantic_vector(reasoning)

            rec = entry.copy()
            rec["llm_vector"] = llm_vec
            final_recs.append(rec)

        path = os.path.join(
            OUTPUT_DIR, f"features_{model.replace(':', '_')}_{split_name}.pt"
        )
        torch.save(final_recs, path)
        print(f"  [OK] Saved -> {path}")

print("\n" + "=" * 60 + "\nCOMPLETE. Now run: python train_fusion.py")
def get_llm_embedding(text: str, model_name: str) -> np.ndarray:
    """Uses Ollama to get the semantic vector for the transcript."""
    payload = {"model": model_name, "prompt": text}
    try:
        response = requests.post(OLLAMA_EMBED_URL, json=payload, timeout=30)
        response.raise_for_status()
        embedding = response.json().get("embedding", [])
        return np.array(embedding, dtype=np.float32)
    except Exception as e:
        print(f"      [!] Ollama Embedding Error ({model_name}): {e}")
        # Return empty array on failure
        return np.zeros(4096, dtype=np.float32) 


from datasets import load_dataset, Audio
...
# ── 3. PROCESS DATASET ────────────────────────────────────────
print("\n[2/3] Streaming dataset and extracting features...")
dataset = load_dataset("stapesai/ssi-speech-emotion-recognition", split="train", streaming=True)
# Ensure audio is always 16kHz for WavLM/Whisper
dataset = dataset.cast_column("file_path", Audio(sampling_rate=16000))

# Data structure to hold our extracted features
extracted_data = {model: [] for model in MODELS}

count = 0
with torch.no_grad():
    for item in dataset:
        if count >= SAMPLE_SIZE:
            break
            
        print(f"\nProcessing Sample {count+1}/{SAMPLE_SIZE}: Emotion={item['emotion']}")
        
        # Access using 'file_path' key instead of 'audio'
        audio_array = np.array(item['file_path']['array'], dtype=np.float32)
        audio_tensor = torch.tensor(audio_array).unsqueeze(0).to(device)
        
        transcript = item['text']
        print(f"  Transcript: \"{transcript}\"")

        # --- A. WavLM Extraction ---
        wavlm_out = wavlm(audio_tensor)
        # Mean pooling over the sequence length to get a single vector per audio
        wavlm_embed = wavlm_out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        
        # --- B. Whisper Extraction ---
        input_features = whisper_processor(audio_array, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        decoder_input_ids = torch.tensor([[1, 1]]) * whisper_model.config.decoder_start_token_id
        decoder_input_ids = decoder_input_ids.to(device)
        whisper_out = whisper_model(input_features, decoder_input_ids=decoder_input_ids)
        # Mean pooling over the sequence
        whisper_embed = whisper_out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

        # --- C. LLM Extraction (Looping through models) ---
        for model_name in MODELS:
            llm_embed = get_llm_embedding(transcript, model_name)
            
            # Combine everything into one giant dictionary
            record = {
                "label": item['emotion'],
                "transcript": transcript,
                "wavlm_vector": wavlm_embed,
                "whisper_vector": whisper_embed,
                "llm_vector": llm_embed
            }
            extracted_data[model_name].append(record)
            
        count += 1

# ── 4. SAVE EXTRACTED FEATURES ────────────────────────────────
print("\n[3/3] Saving extracted feature vectors...")
for model_name in MODELS:
    safe_name = model_name.replace(":", "_")
    save_path = os.path.join(OUTPUT_DIR, f"features_{safe_name}.pt")
    
    # Save as PyTorch binary file (dictionary containing lists of numpy arrays)
    torch.save(extracted_data[model_name], save_path)
    print(f"  Saved -> {save_path}")

print("\n" + "="*60)
print(" EXTRACTION COMPLETE.")
print(" Next Step: We can now train a Scikit-Learn Classifier (e.g. SVM)")
print(" on these saved .pt files to see which LLM's embeddings")
print(" yielded the highest Accuracy and F1 Score!")
print("="*60)

import os
import gc
import time
import torch
import numpy as np
import pandas as pd
import requests
import librosa
import opensmile
from datasets import load_dataset
from transformers import WavLMModel, WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
from tqdm import tqdm

# ── CONFIGURATION ─────────────────────────────────────────────
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
MODELS = ["llama3.1:8b"]
EMBED_MODEL = "nomic-embed-text"
OUTPUT_DIR = "extracted_features"
AUDIO_DIR = "../speech_synthesis_dataset/openvoice_output"
LORA_PATH = "../whisper-dysarthric-lora/checkpoint-500"
SAMPLE_SIZE = 10000 # 10,000 dataset rows * 2 speakers = 20,000 clips!

os.makedirs(OUTPUT_DIR, exist_ok=True)

def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_llm_reasoning(text, model_name):
    prompt = f'Text: "{text}"\nDescribe the likely emotional state of the speaker in 3-5 keywords. Output ONLY the keywords separated by commas.'
    payload = {"model": model_name, "messages": [{"role": "user", "content": prompt}], "stream": False, "options": {"temperature": 0.3}}
    try:
        res = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=60)
        return res.json()["message"]["content"].strip()
    except Exception:
        return "neutral, calm"

def get_semantic_vector(reasoning_text):
    try:
        res = requests.post(OLLAMA_EMBED_URL, json={"model": EMBED_MODEL, "input": reasoning_text}, timeout=30)
        return np.array(res.json()["embeddings"][0], dtype=np.float32)
    except:
        return np.zeros(768, dtype=np.float32)

print("="*60)
print(" DYSARTHRIC PIPELINE v2.0 (FULL 20K CLIPS: F01 + M03)")
print("="*60)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("\n>>> STAGE 1: Loading Models...")
wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(device)

print("[+] Loading Fine-Tuned Whisper Model (LoRA)...")
whisper_proc = WhisperProcessor.from_pretrained("openai/whisper-small")
base_whisper = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)
whisper_model = PeftModel.from_pretrained(base_whisper, LORA_PATH)

smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals)

dataset = load_dataset("stapesai/ssi-speech-emotion-recognition", split="train")
base_data = []

print(f"\n>>> STAGE 2: Processing Audio (Extracting Vectors & Generating Text)...")
with torch.no_grad():
    for i in tqdm(range(min(SAMPLE_SIZE, len(dataset)))):
        item = dataset[i]
        emotion = item["emotion"]
        
        # Loop through both speakers to get all 20,000 clips
        for speaker in ["F01", "M03"]:
            audio_path = os.path.join(AUDIO_DIR, speaker, f"{speaker}_clip_{i}_{emotion}.wav")
            if not os.path.exists(audio_path):
                continue
                
            arr, sr = librosa.load(audio_path, sr=16000)
            
            # 1. WavLM Vector
            ten = torch.tensor(arr).unsqueeze(0).to(device)
            w_out = wavlm(ten).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            
            # 2. Whisper Transcription & Vector
            f_in = whisper_proc(arr, sampling_rate=16000, return_tensors="pt").input_features.to(device)
            
            # Translate dysarthric audio to English Text!
            predicted_ids = whisper_model.generate(f_in, language="english", task="transcribe")
            transcription = whisper_proc.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            # Get Whisper Acoustic Vector
            encoder_outputs = whisper_model.get_encoder()(f_in)
            wh_out = encoder_outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            
            # 3. eGeMAPS Vector
            egemaps_df = smile.process_signal(arr, sr)
            egemaps_vec = egemaps_df.values.flatten().astype(np.float32)
            
            base_data.append({
                "label": emotion,
                "speaker": speaker,
                "transcription": transcription,
                "wavlm_vector": w_out,
                "whisper_vector": wh_out,
                "egemaps_vector": egemaps_vec
            })

print("[!] Unloading acoustic models...")
del wavlm, whisper_model, base_whisper
clear_gpu()

print("\n>>> STAGE 3: LLM Emotional Extraction from Transcriptions...")
for model in MODELS:
    print(f"\nProcessing LLM: {model}")
    final_recs = []
    for entry in tqdm(base_data):
        reasoning = get_llm_reasoning(entry["transcription"], model)
        llm_vec = get_semantic_vector(reasoning)

        rec = entry.copy()
        rec["llm_vector"] = llm_vec
        final_recs.append(rec)

    path = os.path.join(OUTPUT_DIR, f"final_features_{model.replace(':', '_')}.pt")
    torch.save(final_recs, path)
    print(f"  [OK] Saved -> {path}")

print("COMPLETE. Run python train_fusion.py to train and save the SVM.")

"""
Memory-Safe MELD Multimodal Feature Extractor.
Processes in two distinct passes to prevent RTX 3050 (6GB) VRAM overflow.
Pass 1: Acoustic/Linguistic (WavLM, Whisper, openSMILE) -> Checkpoint
Pass 2: Semantic (Ollama: LLaMA + Nomic) -> Final PT
"""

import os
import gc
import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
import requests

try:
    from transformers import WavLMModel, AutoFeatureExtractor, WhisperModel, AutoProcessor
    import opensmile
except ImportError:
    pass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
LLAMA_MODEL = "llama3.1:8b"
EMBED_MODEL = "nomic-embed-text"

def clear_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_llm_reasoning(text):
    prompt = f'Text: "{text}"\nDescribe the likely emotional state of the speaker in 3-5 keywords. Output ONLY the keywords separated by commas.'
    payload = {"model": LLAMA_MODEL, "messages": [{"role": "user", "content": prompt}], "stream": False, "options": {"temperature": 0.3}}
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

def pass_1_acoustic(df, meld_audio_dir, chk_path):
    print("\n" + "="*50)
    print(" PASS 1: ACOUSTIC & LINGUISTIC EXTRACTION (GPU)")
    print("="*50)
    
    if os.path.exists(chk_path):
        data = torch.load(chk_path, weights_only=False)
        print(f"[!] Found Pass 1 checkpoint with {len(data)} records. Skipping Pass 1.")
        return data

    wavlm_ext = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    wavlm_model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(DEVICE).eval()
    whisper_proc = AutoProcessor.from_pretrained("openai/whisper-small")
    whisper_model = WhisperModel.from_pretrained("openai/whisper-small").to(DEVICE).eval()
    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals)

    extracted_data = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Audio Processing"):
        dia_id = row['Dialogue_ID']
        utt_id = row['Utterance_ID']
        audio_file = os.path.join(meld_audio_dir, f"dia{dia_id}_utt{utt_id}.wav")
        if not os.path.exists(audio_file): continue
            
        try:
            waveform, sr = torchaudio.load(audio_path=audio_file)
            if sr != 16000: waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            waveform = waveform.squeeze().numpy()
            
            with torch.no_grad():
                inputs = wavlm_ext(waveform, sampling_rate=16000, return_tensors="pt").to(DEVICE)
                w = wavlm_model(**inputs).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                
                inputs = whisper_proc(waveform, sampling_rate=16000, return_tensors="pt").to(DEVICE)
                wh = whisper_model.encoder(inputs.input_features).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                
            e = smile.process_file(audio_file).values.flatten()
            
            extracted_data.append({
                "utterance_id": f"dia{dia_id}_utt{utt_id}",
                "label": str(row['Emotion']).strip().upper(),
                "transcription": str(row['Utterance']),
                "wavlm_vector": w,
                "whisper_vector": wh,
                "egemaps_vector": e
            })
        except Exception as e:
            pass
            
        if len(extracted_data) % 500 == 0:
            torch.save(extracted_data, chk_path)
            
    torch.save(extracted_data, chk_path)
    
    # CRITICAL: Destroy PyTorch models to free VRAM for Ollama
    del wavlm_model, whisper_model, wavlm_ext, whisper_proc, smile
    clear_vram()
    print("[+] Pass 1 Complete. VRAM Cleared.")
    return extracted_data

def pass_2_semantic(base_data, output_pt):
    print("\n" + "="*50)
    print(" PASS 2: SEMANTIC EXTRACTION (OLLAMA API)")
    print("="*50)
    
    chk_path = output_pt.replace(".pt", "_llm_checkpoint.pt")
    final_recs = []
    
    if os.path.exists(chk_path):
        final_recs = torch.load(chk_path, weights_only=False)
        print(f"[!] Resuming Pass 2 from checkpoint: {len(final_recs)} samples processed.")

    if len(final_recs) >= len(base_data):
        return final_recs
        
    for i in tqdm(range(len(final_recs), len(base_data)), initial=len(final_recs), total=len(base_data), desc="LLM Processing"):
        entry = base_data[i].copy()
        
        reasoning = get_llm_reasoning(entry["transcription"])
        entry["llm_vector"] = get_semantic_vector(reasoning)
        
        final_recs.append(entry)
        
        if len(final_recs) % 100 == 0:
            torch.save(final_recs, chk_path)
            
    torch.save(final_recs, output_pt)
    print(f"[+] Extraction fully complete. Saved {len(final_recs)} records to {output_pt}")
    return final_recs

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    meld_audio_dir = os.path.join(base_dir, "data", "MELD_Audio")
    output_pt = os.path.join(base_dir, "data", "meld_real_features.pt")
    pass1_chk = os.path.join(base_dir, "data", "meld_pass1_checkpoint.pt")
    
    csv_path = os.path.join(meld_audio_dir, "train_sent_emo.csv")
    df = pd.read_csv(csv_path)
    
    # 1. Run Acoustic/Linguistic & free VRAM
    base_data = pass_1_acoustic(df, meld_audio_dir, pass1_chk)
    
    # 2. Run Ollama Semantic
    pass_2_semantic(base_data, output_pt)

if __name__ == "__main__":
    main()

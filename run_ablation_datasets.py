import os
import sys
import gc
import torch
import numpy as np
import pandas as pd
import requests
import librosa
import opensmile
from transformers import WavLMModel, WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
from tqdm import tqdm
import subprocess

# Add current dir to path to import train_fusion
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_fusion import CrossModalAttentionNetwork

import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
MODELS = ["llama3.1:8b"]
EMBED_MODEL = "nomic-embed-text"
LORA_PATH = "../whisper-dysarthric-lora/checkpoint-500"

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

def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def unload_ollama_model(model_name="llama3.1:8b"):
    try:
        print(f"[!] Force-unloading Ollama model {model_name} from VRAM...")
        requests.post(OLLAMA_CHAT_URL.replace("/chat", "/generate"), json={"model": model_name, "keep_alive": 0})
    except Exception as e:
        pass

def extract_features(metadata_path, output_pt_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    unload_ollama_model()
    clear_gpu()
    
    chk_path = output_pt_path.replace(".pt", "_checkpoint.pt")
    base_data = []
    if os.path.exists(chk_path):
        try:
            base_data = torch.load(chk_path, weights_only=False)
            print(f"[!] Resuming acoustic extraction from checkpoint: {len(base_data)} samples already processed.")
        except:
            print("[!] Failed to load checkpoint. Starting from scratch.")
            
    df = pd.read_csv(metadata_path)
    
    if len(base_data) < len(df):
        print(f"\n[+] Loading Models on {device}...")
        wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(device)
        whisper_proc = WhisperProcessor.from_pretrained("openai/whisper-small")
        base_whisper = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)
        try:
            whisper_model = PeftModel.from_pretrained(base_whisper, LORA_PATH)
        except:
            print("[!] Warning: LoRA not found, using base whisper.")
            whisper_model = base_whisper
            
        smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals)
        
        print(f"\n[+] Processing {len(df)} Audio files from {metadata_path}...")
        with torch.no_grad():
            for i in tqdm(range(len(base_data), len(df)), initial=len(base_data), total=len(df)):
                row = df.iloc[i]
                audio_path = row['filepath']
                if not os.path.isabs(audio_path):
                    audio_path = os.path.join("../Dataset_Degradation", audio_path.replace("\\", "/"))
                
                if not os.path.exists(audio_path):
                    continue
                    
                try:
                    arr, sr = librosa.load(audio_path, sr=16000)
                except:
                    continue
                    
                ten = torch.tensor(arr).unsqueeze(0).to(device)
                w_out = wavlm(ten).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                
                f_in = whisper_proc(arr, sampling_rate=16000, return_tensors="pt").input_features.to(device)
                predicted_ids = whisper_model.generate(f_in, language="english", task="transcribe")
                transcription = whisper_proc.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                
                encoder_outputs = whisper_model.get_encoder()(f_in)
                wh_out = encoder_outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                
                egemaps_df = smile.process_signal(arr, sr)
                egemaps_vec = egemaps_df.values.flatten().astype(np.float32)
                
                base_data.append({
                    "label": row["emotion"],
                    "speaker": str(row.get("speaker_id", "UNK")),
                    "transcription": transcription,
                    "wavlm_vector": w_out,
                    "whisper_vector": wh_out,
                    "egemaps_vector": egemaps_vec
                })
                
                if (i + 1) % 250 == 0:
                    torch.save(base_data, chk_path)
            
            torch.save(base_data, chk_path)
            
        print("\n[+] Unloading acoustic models...")
        del wavlm, whisper_model, base_whisper
        clear_gpu()
    else:
        print(f"[+] Acoustic features for all {len(df)} files already extracted.")
        
    print("\n[+] LLM Emotional Extraction from Transcriptions...")
    llm_chk_path = output_pt_path.replace(".pt", "_llm_checkpoint.pt")
    final_recs = []
    if os.path.exists(llm_chk_path):
        try:
            final_recs = torch.load(llm_chk_path, weights_only=False)
            print(f"[!] Resuming LLM extraction from checkpoint: {len(final_recs)} samples processed.")
        except:
            pass

    if len(final_recs) < len(base_data):
        model_name = MODELS[0]
        for i in tqdm(range(len(final_recs), len(base_data)), initial=len(final_recs), total=len(base_data)):
            entry = base_data[i]
            reasoning = get_llm_reasoning(entry["transcription"], model_name)
            llm_vec = get_semantic_vector(reasoning)
            rec = entry.copy()
            rec["llm_vector"] = llm_vec
            final_recs.append(rec)
            
            if (i + 1) % 250 == 0:
                torch.save(final_recs, llm_chk_path)
                
        torch.save(final_recs, llm_chk_path)
        
    torch.save(final_recs, output_pt_path)
    print(f"  [OK] Saved final dataset -> {output_pt_path}")
    return output_pt_path

def train_and_eval(pt_file_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(pt_file_path, weights_only=False)
    
    X_audio, X_text, y_raw = [], [], []
    for item in data:
        wavlm = np.array(item['wavlm_vector']).flatten()
        whisper = np.array(item['whisper_vector']).flatten()
        egemaps = np.array(item['egemaps_vector']).flatten()
        llm_vec = np.array(item['llm_vector']).flatten()
        
        audio_concat = np.concatenate([wavlm, whisper, egemaps])
        X_audio.append(audio_concat)
        X_text.append(llm_vec)
        
        raw_label = str(item['label']).strip().upper()
        label_map = {"HAPPINESS": "HAP", "SADNESS": "SAD", "FEAR": "FEA", "NEUTRAL": "NEU", "FRUSTRATION": "DIS", "ANGRY": "ANG", "ANGER": "ANG"}
        clean_label = label_map.get(raw_label, raw_label)
        y_raw.append(clean_label)
        
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    num_classes = len(le.classes_)
    
    X_audio = np.array(X_audio)
    X_text = np.array(X_text)
    
    idx = np.arange(len(y))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
    
    audio_scaler = StandardScaler()
    text_scaler = StandardScaler()
    X_audio_train = audio_scaler.fit_transform(X_audio[idx_train])
    X_audio_test = audio_scaler.transform(X_audio[idx_test])
    X_text_train = text_scaler.fit_transform(X_text[idx_train])
    X_text_test = text_scaler.transform(X_text[idx_test])
    
    train_dataset = TensorDataset(torch.FloatTensor(X_audio_train), torch.FloatTensor(X_text_train), torch.LongTensor(y[idx_train]))
    test_dataset = TensorDataset(torch.FloatTensor(X_audio_test), torch.FloatTensor(X_text_test), torch.LongTensor(y[idx_test]))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    audio_dim = X_audio.shape[1]
    text_dim = X_text.shape[1]
    model = CrossModalAttentionNetwork(audio_dim, text_dim, hidden_dim=256, num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(25):
        model.train()
        for a_b, t_b, y_b in train_loader:
            a_b, t_b, y_b = a_b.to(device), t_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            preds = model(a_b, t_b)
            loss = criterion(preds, y_b)
            loss.backward()
            optimizer.step()
            
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for a_b, t_b, y_b in test_loader:
            a_b, t_b, y_b = a_b.to(device), t_b.to(device), y_b.to(device)
            preds = model(a_b, t_b).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_b.cpu().numpy())
            
    from sklearn.metrics import accuracy_score
    return accuracy_score(all_targets, all_preds)

if __name__ == "__main__":
    results_file = "dataset_ablation_results.txt"
    with open(results_file, "w") as f:
        f.write("Dataset Ablation Study Results\n")
        f.write("==============================\n")
        
    datasets_to_run = [
        {"name": "Pure DSP", "script": "DysarthricDegrader.py", "meta": "pure_dsp_dataset/metadata.csv"},
        {"name": "CORAL Aligned", "script": "coral_degrader.py", "meta": "coral_dataset/metadata.csv"},
        {"name": "Vowel Segmentation", "script": "vowel_segment_degrader.py", "meta": "vowel_dataset/metadata.csv"}
    ]
    
    # 1. Generate All Datasets First
    print(f"\n{'='*50}\nPHASE 1: GENERATING ALL 10K DATASETS\n{'='*50}")
    for ds in datasets_to_run:
        print(f"[1] Running {ds['script']} to generate {ds['name']} dataset...")
        cwd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../Dataset_Degradation")
        subprocess.run([sys.executable, ds['script']], cwd=cwd)
        
    print(f"\n{'='*50}\nPHASE 2: RUNNING ARCHITECTURE ON DATASETS\n{'='*50}")
    for ds in datasets_to_run:
        cwd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../Dataset_Degradation")
        metadata_path = os.path.join(cwd, ds['meta'])
        if not os.path.exists(metadata_path):
            print(f"[!] ERROR: {metadata_path} not found after running script. Skipping.")
            continue
            
        # 2. Extract Features
        pt_path = f"extracted_features/ablation_{ds['name'].replace(' ', '_')}.pt"
        print(f"[2] Extracting multimodal features from metadata...")
        extract_features(metadata_path, pt_path)
        
        # 3. Train and Evaluate
        print(f"[3] Training Cross-Attention Network on {ds['name']}...")
        acc = train_and_eval(pt_path)
        print(f"*** FINAL ACCURACY ({ds['name']}): {acc:.2%} ***")
        
        # 4. Save to file
        with open(results_file, "a") as f:
            f.write(f"{ds['name']} Dataset -> Cross-Attention Accuracy: {acc:.2%}\n")
            
    print(f"\n[+] All datasets evaluated! Results saved in {results_file}")

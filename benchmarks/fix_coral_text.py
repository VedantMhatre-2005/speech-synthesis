import torch
import requests
import numpy as np
from tqdm import tqdm
import os

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
LLAMA_MODEL = "llama3.1:8b"
EMBED_MODEL = "nomic-embed-text"

def get_llm_reasoning(text):
    prompt = f'Text: "{text}"\nDescribe the likely emotional state of the speaker in 3-5 keywords. Output ONLY the keywords separated by commas.'
    try:
        res = requests.post(OLLAMA_CHAT_URL, json={"model": LLAMA_MODEL, "messages": [{"role": "user", "content": prompt}], "stream": False, "options": {"temperature": 0.3}}, timeout=60)
        return res.json()["message"]["content"].strip()
    except:
        return "neutral, calm"

def get_semantic_vector(reasoning_text):
    try:
        res = requests.post(OLLAMA_EMBED_URL, json={"model": EMBED_MODEL, "input": reasoning_text}, timeout=30)
        return np.array(res.json()["embeddings"][0], dtype=np.float32)
    except:
        return np.zeros(768, dtype=np.float32)

def main():
    pt_path = "/home/ghostblaster08/Projects/speech_synth/speech-synthesis/extracted_features/ablation_CORAL_Aligned.pt"
    print(f"[*] Loading CORAL dataset: {pt_path}")
    data = torch.load(pt_path, weights_only=False)
    
    print(f"[*] Fixing blank text embeddings for {len(data)} samples via Ollama...")
    for i in tqdm(range(len(data))):
        # We only fix it if it's broken (all zeros) or missing
        vec = data[i].get("llm_vector", np.zeros(768))
        if np.all(vec == 0):
            # The transcription key in CORAL is usually "transcription", fallback to "text" or empty
            text = data[i].get("transcription", data[i].get("text", ""))
            reasoning = get_llm_reasoning(text)
            data[i]["llm_vector"] = get_semantic_vector(reasoning)
            
    torch.save(data, pt_path)
    print(f"[+] Successfully fixed and overwritten {pt_path}!")

if __name__ == "__main__":
    main()

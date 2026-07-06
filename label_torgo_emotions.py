import os
import json
import requests
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# ── CONFIGURATION ─────────────────────────────────────────────
OLLAMA_CHAT_URL  = "http://localhost:11434/api/chat"
MODEL            = "llama3.1:8b" 
OUTPUT_CSV       = "torgo_emotion_labels.csv"

def get_llm_emotion(text):
    """Asks the LLM to deduce the emotional state of the speaker from the text."""
    # Handle empty or single character readings in TORGO
    if pd.isna(text) or len(text.strip()) <= 1:
        return "neutral"

    prompt = (
        f"You are analyzing transcripts of spoken text to determine the emotion.\n"
        f"Text: \"{text}\"\n"
        "Describe the likely emotional state of the speaker in 1-2 keywords (e.g., happy, sad, frustrated, neutral, angry). "
        "Output ONLY the keywords, nothing else."
    )
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.3}
    }
    try:
        res = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=60)
        return res.json()["message"]["content"].strip().lower()
    except Exception as e:
        return "neutral"

def main():
    print("Loading TORGO dataset from HuggingFace...")
    # Loading the dataset from HuggingFace community upload
    # Note: TORGO is large. You may want to stream it or slice it.
    dataset = load_dataset("abnerh/TORGO-database", split="train") 
    
    print(f"Loaded {len(dataset)} samples. Starting emotion labeling via LLM...")
    
    labeled_data = []
    
    # Using a small subset for demonstration. Remove '[:100]' to run on the full dataset.
    # Note: Running full dataset will take significant time with local LLM.
    subset = dataset.select(range(100)) 
    
    for item in tqdm(subset, desc="Labeling Emotions"):
        text = item.get("text", "")
        speaker = item.get("speaker_id", "")
        
        # Determine emotion
        emotion = get_llm_emotion(text)
        
        labeled_data.append({
            "speaker_id": speaker,
            "text": text,
            "inferred_emotion": emotion,
            "audio_file": item.get("audio", {}).get("path", "")
        })
    
    df = pd.DataFrame(labeled_data)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df)} labeled samples to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

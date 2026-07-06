
import sys
import os
import time
import re
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import classification_report, accuracy_score
import requests

# Add current dir to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents import OllamaMistralAgent

# ── CONFIGURATION ─────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/chat"
MODELS     = ["qwen2.5:7b", "qwen3:14B", "llama3.1:8b"]
SAMPLE_SIZE = 20  # Samples per model for quick comparison

# Mapping dataset labels to our system's expected tags
# SSI Labels: ANG, DIS, FEA, HAP, NEU, SAD, CAL, SUR
LABEL_MAPPING = {
    "ANG": "Angry",
    "HAP": "Happy",
    "NEU": "Neutral",
    "SAD": "Sad",
    "CAL": "Neutral", # Calm -> Neutral
    # DIS, FEA, SUR are skipped or mapped to closest
}

def clean_label(label_str):
    return LABEL_MAPPING.get(label_str, None)

def extract_emotion(response):
    match = re.search(r"\[(.*?)\]", response)
    if match:
        return match.group(1).strip().capitalize()
    return "Neutral" # Default

# ── BENCHMARK RUNNER ──────────────────────────────────────────

def run_benchmark():
    print("="*60)
    print(" AAC Multi-Model Emotion Benchmark")
    print(f" Dataset: stapesai/ssi-speech-emotion-recognition")
    print("="*60)

    # 1. Load Dataset
    print("\n[  ] Loading dataset from Hugging Face ...")
    dataset = load_dataset("stapesai/ssi-speech-emotion-recognition", split="train", streaming=True)
    
    samples = []
    count = 0
    for item in dataset:
        mapped_label = clean_label(item['emotion'])
        if mapped_label:
            samples.append({
                "text": item['text'],
                "expected": mapped_label
            })
            count += 1
        if count >= SAMPLE_SIZE:
            break
    
    print(f"[OK] Loaded {len(samples)} valid samples.")

    # 2. Iterate Models
    results = []

    for model_name in MODELS:
        print(f"\n>>> Benchmarking Model: {model_name}")
        y_true = []
        y_pred = []
        latencies = []

        # Mock context for the agent
        mock_context = "Patient: Aarav. Context: General interaction."
        
        for i, sample in enumerate(samples):
            print(f"    [{i+1}/{SAMPLE_SIZE}] Input: \"{sample['text']}\"")
            
            start_time = time.time()
            
            # Call Ollama directly for benchmarking to avoid agent history overhead
            payload = {
                "model":    model_name,
                "messages": [
                    {"role": "system", "content": f"You are an AAC assistant. Output ONLY the emotion in brackets and the phrase. Format: [Emotion] Phrase. Allowed: [Happy], [Sad], [Angry], [Neutral]. Input: {sample['text']}"},
                    {"role": "user", "content": sample['text']}
                ],
                "stream":   False,
                "options":  {"temperature": 0, "num_predict": 100} 
            }

            try:
                response = requests.post(OLLAMA_URL, json=payload, timeout=60)
                response.raise_for_status()
                res_content = response.json()["message"]["content"]
                pred_emotion = extract_emotion(res_content)
            except Exception as e:
                print(f"      ✗ Error: {e}")
                pred_emotion = "Error"

            latency = time.time() - start_time
            
            y_true.append(sample['expected'])
            y_pred.append(pred_emotion)
            latencies.append(latency)
            
            print(f"      → Expected: {sample['expected']} | Predicted: {pred_emotion} ({latency:.2f}s)")

        # 3. Calculate Metrics
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        avg_latency = sum(latencies) / len(latencies)
        
        results.append({
            "Model": model_name,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": report['weighted avg']['precision'],
            "Recall": report['weighted avg']['recall'],
            "F1-Score": report['weighted avg']['f1-score'],
            "Avg Latency": avg_latency
        })

    # 4. Final Comparison Table
    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print(" FINAL COMPARATIVE ANALYSIS")
    print("="*80)
    print(df.to_markdown(index=False))
    print("="*80)

if __name__ == "__main__":
    run_benchmark()

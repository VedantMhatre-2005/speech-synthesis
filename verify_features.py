import torch
import numpy as np
import os
MODELS = ["qwen2.5_7b", "llama3.1_8b", "mistral_7b"]
DIR = "extracted_features"
def verify():
    print("="*60)
    print(" FEATURE VECTOR VERIFICATION")
    print("="*60)
    for m in MODELS:
        path = os.path.join(DIR, f"features_{m}.pt")
        if not os.path.exists(path):
            print(f"[!] {m}: File not found.")
            continue
        data = torch.load(path, weights_only=False)
        print(f"\n>>> Model: {m}")
        print(f"    Total Samples: {len(data)}")
        sample = data[0]
        # Check WavLM
        w_vec = np.array(sample['wavlm_vector'])
        print(f"    WavLM Vector   : Shape={w_vec.shape}, Mean={w_vec.mean():.4f}, Std={w_vec.std():.4f}")
        # Check LLM
        l_vec = np.array(sample['llm_vector'])
        print(f"    LLM Vector     : Shape={l_vec.shape}, Mean={l_vec.mean():.4f}, Std={l_vec.std():.4f}")
        if np.all(l_vec == 0):
            print("    [!!!] CRITICAL: LLM vector is ALL ZEROS.")
        elif l_vec.shape[0] < 10:
            print("    [!!!] CRITICAL: LLM vector is too small (likely a fallback).")
    print("\n" + "="*60)
if __name__ == "__main__":
    verify()
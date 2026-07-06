import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

MODELS = ["llama3.1:8b"]
OUTPUT_DIR = "extracted_features"

def train_and_evaluate():
    print("="*80)
    print(" DYSARTHRIC SVM FUSION TRAINING (With vs Without eGeMAPS)")
    print("="*80)
    
    for model_name in MODELS:
        file_path = os.path.join(OUTPUT_DIR, f"features_{model_name.replace(':', '_')}_dysarthric.pt")
        
        if not os.path.exists(file_path):
            print(f"[!] Warning: {file_path} not found. Did you run benchmark_multimodal.py?")
            return
            
        data = torch.load(file_path, weights_only=False)
        print(f"\n>>> Loaded {len(data)} dysarthric feature records.")
        
        X_without_egemaps = []
        X_with_egemaps = []
        y_raw = []
        
        for item in data:
            wavlm_vec = np.array(item['wavlm_vector']).flatten()
            whisper_vec = np.array(item['whisper_vector']).flatten()
            llm_vec = np.array(item['llm_vector']).flatten()
            egemaps_vec = np.array(item['egemaps_vector']).flatten()
            
            # Baseline Pipeline (Without eGeMAPS)
            vec_no_ege = np.concatenate([wavlm_vec, whisper_vec, llm_vec])
            X_without_egemaps.append(vec_no_ege)
            
            # Upgraded Pipeline (With eGeMAPS)
            vec_with_ege = np.concatenate([wavlm_vec, whisper_vec, llm_vec, egemaps_vec])
            X_with_egemaps.append(vec_with_ege)
            
            y_raw.append(item['label'])
            
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        
        # We'll use the exact same splits for both experiments
        idx = np.arange(len(y))
        idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
        y_train, y_test = y[idx_train], y[idx_test]
        
        def run_experiment(name, X):
            X = np.array(X)
            X_train, X_test = X[idx_train], X[idx_test]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            pca = PCA(n_components=min(150, len(X_train)), random_state=42)
            X_train_pca = pca.fit_transform(X_train_scaled)
            X_test_pca = pca.transform(X_test_scaled)
            
            clf = LinearSVC(class_weight='balanced', max_iter=5000, random_state=42, dual=False)
            clf.fit(X_train_pca, y_train)
            
            y_pred = clf.predict(X_test_pca)
            acc = accuracy_score(y_test, y_pred)
            return acc
            
        # Run Baseline
        print("\n[+] Training Old Pipeline (WavLM + Whisper + LLM) on Dysarthric Data...")
        acc_baseline = run_experiment("Baseline", X_without_egemaps)
        print(f"    --> Accuracy without eGeMAPS: {acc_baseline:.2%}")
        
        # Run Upgraded
        print("\n[+] Training New Pipeline (WavLM + Whisper + LLM + eGeMAPS) on Dysarthric Data...")
        acc_upgraded = run_experiment("Upgraded", X_with_egemaps)
        print(f"    --> Accuracy with eGeMAPS:    {acc_upgraded:.2%}")
        
        print("\n" + "="*80)
        print(f" OVERALL IMPROVEMENT: +{(acc_upgraded - acc_baseline)*100:.2f}%")
        print("="*80)

if __name__ == "__main__":
    train_and_evaluate()
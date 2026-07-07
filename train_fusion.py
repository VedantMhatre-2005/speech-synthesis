import os
import sys
import torch
import joblib
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

MODELS = ["llama3.1:8b"]
OUTPUT_DIR = "extracted_features"
MODEL_SAVE_DIR = "saved_models"

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

def train_and_evaluate():
    print("="*80)
    print(" TRAINING AND SAVING ULTIMATE DYSARTHRIC SVM CLASSIFIER")
    print("="*80)
    
    for model_name in MODELS:
        file_path = os.path.join(OUTPUT_DIR, f"final_features_{model_name.replace(':', '_')}.pt")
        
        if not os.path.exists(file_path):
            print(f"[!] Warning: {file_path} not found. Did you run benchmark_multimodal.py?")
            return
            
        data = torch.load(file_path, weights_only=False)
        print(f"\n>>> Loaded {len(data)} dysarthric feature records.")
        
        X = []
        y_raw = []
        
        for item in data:
            wavlm_vec = np.array(item['wavlm_vector']).flatten()
            whisper_vec = np.array(item['whisper_vector']).flatten()
            llm_vec = np.array(item['llm_vector']).flatten()
            egemaps_vec = np.array(item['egemaps_vector']).flatten()
            
            # The Ultimate Concatenated Vector
            vec = np.concatenate([wavlm_vec, whisper_vec, llm_vec, egemaps_vec])
            X.append(vec)
            y_raw.append(item['label'])
            
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        
        idx = np.arange(len(y))
        idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
        
        X = np.array(X)
        X_train, X_test = X[idx_train], X[idx_test]
        y_train, y_test = y[idx_train], y[idx_test]
        
        print("[+] Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print("[+] Applying PCA Dimensionality Reduction...")
        pca = PCA(n_components=min(150, len(X_train)), random_state=42)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        print("[+] Training SVM...")
        clf = LinearSVC(class_weight='balanced', max_iter=5000, random_state=42, dual=False)
        clf.fit(X_train_pca, y_train)
        
        y_pred = clf.predict(X_test_pca)
        acc = accuracy_score(y_test, y_pred)
        print(f"    --> Final Model Accuracy: {acc:.2%}")
        
        # --- SAVING THE PIPELINE FOR PRODUCTION USE ---
        print("\n[+] Saving Models to Disk...")
        joblib.dump(le, os.path.join(MODEL_SAVE_DIR, 'label_encoder.pkl'))
        joblib.dump(scaler, os.path.join(MODEL_SAVE_DIR, 'scaler.pkl'))
        joblib.dump(pca, os.path.join(MODEL_SAVE_DIR, 'pca.pkl'))
        joblib.dump(clf, os.path.join(MODEL_SAVE_DIR, 'svm_classifier.pkl'))
        
        print(f"  [OK] Pipeline saved in '{MODEL_SAVE_DIR}/'. Ready for Voice Gen!")
        print("="*80)

if __name__ == "__main__":
    train_and_evaluate()
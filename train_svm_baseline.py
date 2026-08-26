import os
import torch
import joblib
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

MODELS = ["llama3.1:8b"]
OUTPUT_DIR = "extracted_features"

def eval_svm_baseline():
    print("="*80)
    print(" TRAINING & EVALUATING BASELINE SVM (For Comparison)")
    print("="*80)
    
    file_path = os.path.join(OUTPUT_DIR, f"final_features_{MODELS[0].replace(':', '_')}.pt")
    if not os.path.exists(file_path):
        print(f"[!] Warning: {file_path} not found.")
        return
        
    data = torch.load(file_path, weights_only=False)
    
    X = []
    y_raw = []
    
    for item in data:
        wavlm_vec = np.array(item['wavlm_vector']).flatten()
        whisper_vec = np.array(item['whisper_vector']).flatten()
        llm_vec = np.array(item['llm_vector']).flatten()
        egemaps_vec = np.array(item['egemaps_vector']).flatten()
        
        # The Ultimate Concatenated Vector for SVM
        vec = np.concatenate([wavlm_vec, whisper_vec, llm_vec, egemaps_vec])
        X.append(vec)
        
        # Label Standardization (Same as DL model!)
        raw_label = item['label'].strip().upper()
        label_map = {
            "HAPPINESS": "HAP",
            "SADNESS": "SAD",
            "FEAR": "FEA",
            "NEUTRAL": "NEU",
            "FRUSTRATION": "DIS",
            "ANGRY": "ANG",
            "ANGER": "ANG"
        }
        clean_label = label_map.get(raw_label, raw_label)
        y_raw.append(clean_label)
        
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
    
    print(f"\n    --> Final SVM Accuracy: {acc:.2%}")
    print("\n    --> DETAILED CLINICAL METRICS (Precision, Recall/Sensitivity, F1)")
    target_names = [str(c) for c in le.classes_]
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("="*80)

if __name__ == "__main__":
    eval_svm_baseline()

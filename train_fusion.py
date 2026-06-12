import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

# Add current dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

MODELS = ["llama3.1:8b"]
OUTPUT_DIR = "extracted_features"

def load_split(model_name, split):
    safe_name = model_name.replace(":", "_")
    file_path = os.path.join(OUTPUT_DIR, f"features_{safe_name}_{split}.pt")
    
    if not os.path.exists(file_path):
        print(f"[!] Warning: {file_path} not found.")
        return None, None
        
    data = torch.load(file_path, weights_only=False)
    
    X = []
    y = []
    
    for item in data:
        wavlm_vec = np.array(item['wavlm_vector']).flatten()
        whisper_vec = np.array(item['whisper_vector']).flatten()
        llm_vec = np.array(item['llm_vector']).flatten()
        
        fused_vector = np.concatenate([wavlm_vec, whisper_vec, llm_vec])
        
        X.append(fused_vector)
        y.append(item['label'])
        
    return np.array(X), np.array(y)

def train_and_evaluate():
    print("="*80)
    print(" MULTIMODAL FUSION LAYER: FULL DATASET TRAINING")
    print("="*80)
    
    results = []
    
    for model_name in MODELS:
        print(f"\n>>> Training Fusion Model using LLM semantics from: {model_name}")
        
        # Load Train Data
        X_train_raw, y_train_raw = load_split(model_name, "train")
        # Load Test Data
        X_test_raw, y_test_raw = load_split(model_name, "test")
        
        if X_train_raw is None or X_test_raw is None:
            continue
            
        print(f"  [+] Loaded {len(X_train_raw)} Train samples, {len(X_test_raw)} Test samples.")
        
        # Encode labels (ANG, HAP, NEU, etc.)
        le = LabelEncoder()
        y_train = le.fit_transform(y_train_raw)
        
        # Handle cases where test set might have classes not in train set
        # (Very rare on 10k train, but safe to do)
        y_test = []
        valid_test_idx = []
        for idx, val in enumerate(y_test_raw):
            if val in le.classes_:
                y_test.append(le.transform([val])[0])
                valid_test_idx.append(idx)
        
        y_test = np.array(y_test)
        X_test_raw = X_test_raw[valid_test_idx]

        # 1. Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)
        
        # 2. PCA: Dimensionality Reduction (Crucial for high-dim vectors)
        # We reduce the ~2048 dimensions down to the 100 most important mathematical components.
        # This removes noise and prevents the SVM from getting "confused"
        print("  [+] Applying PCA (Compressing dimensions...)")
        n_components = min(150, len(X_train_raw)) # Max 150 components or sample size
        pca = PCA(n_components=n_components, random_state=42)
        X_train = pca.fit_transform(X_train_scaled)
        X_test = pca.transform(X_test_scaled)
        
        # 3. Train SVM with Balanced Class Weights
        # 'balanced' forces the model to care equally about rare emotions (like 'Surprise')
        # and common emotions (like 'Neutral').
        print("  [+] Training SVM (with balanced weights)...")
        classifier = LinearSVC(class_weight='balanced', max_iter=5000, random_state=42, dual=False)
        classifier.fit(X_train, y_train)
        
        # 4. Predict & Evaluate
        print("  [+] Evaluating on Test Split...")
        y_pred = classifier.predict(X_test)
        
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        results.append({
            "LLM Base": model_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": report['weighted avg']['precision'],
            "Recall": report['weighted avg']['recall'],
            "F1-Score": report['weighted avg']['f1-score']
        })
        
        print(f"\n--- Detailed Report for {model_name} ---")
        print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    if results:
        df = pd.DataFrame(results)
        print("\n" + "="*80)
        print(" FINAL FUSION LAYER COMPARATIVE ANALYSIS (10k Train / 163 Test)")
        print("="*80)
        print(df.to_markdown(index=False))
        print("="*80)

if __name__ == "__main__":
    train_and_evaluate()
"""
train_fusion.py — Context-Engineered SVM Training Pipeline (v2.1)

Changes from v2.0:
  • Reads context-engineered feature vectors (*_ctx.pt) produced by benchmark_multimodal.py v2.1
  • Falls back to legacy (*_plain.pt) vectors if context-engineered ones are not found.
  • GridSearchCV over LinearSVC `C` parameter to squeeze extra accuracy.
  • Per-context accuracy breakdown printed after evaluation.
  • Confusion matrix saved as a CSV for analysis.
  • Saves the final pipeline (scaler, PCA, SVM, LabelEncoder) to saved_models/.

Run:
    python train_fusion.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV

MODELS         = ["llama3.1:8b"]
OUTPUT_DIR     = "extracted_features"
MODEL_SAVE_DIR = "saved_models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ── 1. CROSS-ATTENTION NEURAL NETWORK (Inspired by MulT) ──
class CrossModalAttentionNetwork(nn.Module):
    def __init__(self, audio_dim, text_dim, hidden_dim=256, num_classes=5):
        super().__init__()
        # Project both modalities to a shared latent space
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Cross-Attention: Text asks (Query), Audio answers (Key, Value)
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        
        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, audio_feat, text_feat):
        # audio_feat: (Batch, AudioDim), text_feat: (Batch, TextDim)
        # Project to (Batch, 1, HiddenDim) for attention
        a_proj = self.audio_proj(audio_feat).unsqueeze(1)
        t_proj = self.text_proj(text_feat).unsqueeze(1)
        
        # Cross Attention: Query=Text, Key=Audio, Value=Audio
        attn_out, _ = self.cross_attention(query=t_proj, key=a_proj, value=a_proj)
        
        # Pool and Concatenate the dynamically attended features
        attn_out = attn_out.squeeze(1)
        t_proj = t_proj.squeeze(1)
        fused = torch.cat([attn_out, t_proj], dim=1)
        
        return self.classifier(fused)

def train_and_evaluate():
    print("="*80)
    print(" TRAINING CROSS-ATTENTION DEEP LEARNING MODEL (MulT-Lite)")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for model_name in MODELS:
        file_path = os.path.join(OUTPUT_DIR, f"final_features_{model_name.replace(':', '_')}.pt")
        if not os.path.exists(file_path):
            print(f"[!] Warning: {file_path} not found. Did you run benchmark_multimodal.py?")
            return
            
        data = torch.load(file_path, weights_only=False)
        print(f"\n>>> Loaded {len(data)} dysarthric feature records.")
        
        X_audio, X_text, y_raw = [], [], []
        
        for item in data:
            wavlm = np.array(item['wavlm_vector']).flatten()
            whisper = np.array(item['whisper_vector']).flatten()
            egemaps = np.array(item['egemaps_vector']).flatten()
            llm_vec = np.array(item['llm_vector']).flatten()
            
            audio_concat = np.concatenate([wavlm, whisper, egemaps])
            X_audio.append(audio_concat)
            X_text.append(llm_vec)
            
            # Label Standardization
            raw_label = item['label'].strip().upper()
            label_map = {
                "HAPPINESS": "HAP",
                "SADNESS": "SAD",
                "FEAR": "FEA",
                "NEUTRAL": "NEU",
                "FRUSTRATION": "DIS", # Mapping frustration to disgust/displeasure
                "ANGRY": "ANG",
                "ANGER": "ANG"
            }
            clean_label = label_map.get(raw_label, raw_label)
            y_raw.append(clean_label)
            
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        num_classes = len(le.classes_)
        
        X_audio = np.array(X_audio)
        X_text = np.array(X_text)
        
        # Train/Test Split
        idx = np.arange(len(y))
        idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
        
        # Scaling
        print("[+] Scaling features...")
        audio_scaler = StandardScaler()
        text_scaler = StandardScaler()
        
        X_audio_train = audio_scaler.fit_transform(X_audio[idx_train])
        X_audio_test = audio_scaler.transform(X_audio[idx_test])
        
        X_text_train = text_scaler.fit_transform(X_text[idx_train])
        X_text_test = text_scaler.transform(X_text[idx_test])
        
        # PyTorch DataLoaders
        train_dataset = TensorDataset(torch.FloatTensor(X_audio_train), torch.FloatTensor(X_text_train), torch.LongTensor(y[idx_train]))
        test_dataset = TensorDataset(torch.FloatTensor(X_audio_test), torch.FloatTensor(X_text_test), torch.LongTensor(y[idx_test]))
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64)
        
        # Initialize Model
        audio_dim = X_audio.shape[1]
        text_dim = X_text.shape[1]
        model = CrossModalAttentionNetwork(audio_dim, text_dim, hidden_dim=256, num_classes=num_classes).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        print(f"[+] Training PyTorch Model (AudioDim={audio_dim}, TextDim={text_dim})...")
        epochs = 25
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for a_b, t_b, y_b in train_loader:
                a_b, t_b, y_b = a_b.to(device), t_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                preds = model(a_b, t_b)
                loss = criterion(preds, y_b)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            if (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")
                
        # Evaluate
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for a_b, t_b, y_b in test_loader:
                a_b, t_b, y_b = a_b.to(device), t_b.to(device), y_b.to(device)
                preds = model(a_b, t_b).argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_b.cpu().numpy())
                
        # Calculate full metrics
        from sklearn.metrics import classification_report, accuracy_score
        acc = accuracy_score(all_targets, all_preds)
        print(f"\n    --> Final Cross-Attention Accuracy: {acc:.2%}")
        
        print("\n    --> DETAILED CLINICAL METRICS (Precision, Recall/Sensitivity, F1)")
        target_names = [str(c) for c in le.classes_]
        print(classification_report(all_targets, all_preds, target_names=target_names))
        
        # Save Pipeline
        print("\n[+] Saving PyTorch Model and Scalers...")
        joblib.dump(le, os.path.join(MODEL_SAVE_DIR, 'label_encoder.pkl'))
        joblib.dump(audio_scaler, os.path.join(MODEL_SAVE_DIR, 'audio_scaler.pkl'))
        joblib.dump(text_scaler, os.path.join(MODEL_SAVE_DIR, 'text_scaler.pkl'))
        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, 'cross_attention_model.pth'))
        
        print(f"  [OK] Saved to '{MODEL_SAVE_DIR}/'. Ready for main.py integration!")
        print("="*80)

def load_feature_file(model_name: str) -> list:
    """
    Try to load context-engineered vectors first (*_ctx.pt).
    Falls back to plain vectors (*plain*.pt or the original naming) if not found.
    """
    base = model_name.replace(":", "_")

    # Priority 1: new context-engineered file
    ctx_path = os.path.join(OUTPUT_DIR, f"final_features_{base}_ctx.pt")
    if os.path.exists(ctx_path):
        print(f"[+] Loading context-engineered features: {ctx_path}")
        return torch.load(ctx_path, weights_only=False), True

    # Priority 2: original file (non-context-engineered)
    plain_path = os.path.join(OUTPUT_DIR, f"final_features_{base}.pt")
    if os.path.exists(plain_path):
        print(f"[!] Context-engineered file not found. Loading plain features: {plain_path}")
        print(
            "    TIP: Run benchmark_multimodal.py first to generate richer vectors."
        )
        return torch.load(plain_path, weights_only=False), False

    return None, False


def train_and_evaluate():
    print("=" * 80)
    print(" TRAINING CONTEXT-ENGINEERED DYSARTHRIC SVM CLASSIFIER (v2.1)")
    print("=" * 80)

    for model_name in MODELS:
        data, is_context_engineered = load_feature_file(model_name)

        if data is None:
            print(
                f"\n[!] No feature file found for '{model_name}'.\n"
                "    Run benchmark_multimodal.py first."
            )
            return

        print(
            f"\n>>> Loaded {len(data)} feature records. "
            f"Context-engineered: {'YES ✓' if is_context_engineered else 'NO (plain fallback)'}"
        )

        # ── BUILD FEATURE MATRIX ──────────────────────────────────────────────
        X, y_raw, scenarios = [], [], []

        for item in data:
            wavlm_vec   = np.array(item["wavlm_vector"]).flatten()
            whisper_vec = np.array(item["whisper_vector"]).flatten()
            llm_vec     = np.array(item["llm_vector"]).flatten()
            egemaps_vec = np.array(item["egemaps_vector"]).flatten()

            # Concatenated fusion vector
            vec = np.concatenate([wavlm_vec, whisper_vec, llm_vec, egemaps_vec])
            X.append(vec)
            y_raw.append(item["label"])

            # Record scenario for per-context breakdown (may be missing in v2.0 files)
            scenarios.append(item.get("scenario", "unknown"))

        X         = np.array(X, dtype=np.float32)
        scenarios = np.array(scenarios)

        le = LabelEncoder()
        y  = le.fit_transform(y_raw)

        # ── TRAIN / TEST SPLIT ────────────────────────────────────────────────
        idx = np.arange(len(y))
        idx_train, idx_test = train_test_split(
            idx, test_size=0.2, random_state=42, stratify=y
        )

        X_train, X_test     = X[idx_train],         X[idx_test]
        y_train, y_test     = y[idx_train],         y[idx_test]
        sc_train, sc_test   = scenarios[idx_train], scenarios[idx_test]

        # ── SCALING + PCA ─────────────────────────────────────────────────────
        print("[+] Scaling features (StandardScaler) ...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        n_components = min(200, len(X_train))   # bumped from 150 to 200
        print(f"[+] Applying PCA (n_components={n_components}) ...")
        pca = PCA(n_components=n_components, random_state=42)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca  = pca.transform(X_test_scaled)

        explained = pca.explained_variance_ratio_.sum()
        print(f"    Explained variance: {explained:.1%}")

        # ── GRIDSEARCHCV OVER C ───────────────────────────────────────────────
        print("[+] Running GridSearchCV to find optimal LinearSVC C ...")
        param_grid = {"C": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]}
        base_clf   = LinearSVC(
            class_weight="balanced",
            max_iter=10000,
            random_state=42,
            dual=False,
        )
        grid_search = GridSearchCV(
            base_clf,
            param_grid,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
            verbose=1,
        )
        grid_search.fit(X_train_pca, y_train)

        best_C   = grid_search.best_params_["C"]
        best_cv  = grid_search.best_score_
        print(f"    Best C={best_C}  |  CV accuracy={best_cv:.2%}")

        # ── FINAL TRAINING WITH BEST C ────────────────────────────────────────
        print(f"[+] Training final LinearSVC (C={best_C}) ...")
        clf = LinearSVC(
            C=best_C,
            class_weight="balanced",
            max_iter=10000,
            random_state=42,
            dual=False,
        )
        clf.fit(X_train_pca, y_train)

        y_pred = clf.predict(X_test_pca)
        acc    = accuracy_score(y_test, y_pred)
        print(f"\n  ──> Final Test Accuracy : {acc:.2%}")

        # ── FULL CLASSIFICATION REPORT ────────────────────────────────────────
        class_names = le.classes_
        report_str  = classification_report(
            y_test, y_pred, target_names=class_names, zero_division=0
        )
        print("\n── CLASSIFICATION REPORT ─────────────────────────────────────────")
        print(report_str)

        # ── CONFUSION MATRIX ──────────────────────────────────────────────────
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        cm_path = os.path.join(MODEL_SAVE_DIR, "confusion_matrix.csv")
        cm_df.to_csv(cm_path)
        print(f"── CONFUSION MATRIX (saved → {cm_path}) ──────────────────────────")
        print(cm_df.to_string())

        # ── PER-CONTEXT ACCURACY BREAKDOWN ───────────────────────────────────
        unique_scenarios = sorted(set(sc_test))
        if len(unique_scenarios) > 1 or unique_scenarios[0] != "unknown":
            print("\n── PER-CONTEXT ACCURACY ──────────────────────────────────────────")
            context_results = []
            for sc in unique_scenarios:
                mask      = sc_test == sc
                if mask.sum() == 0:
                    continue
                sc_acc    = accuracy_score(y_test[mask], y_pred[mask])
                sc_count  = mask.sum()
                context_results.append(
                    {"Scenario": sc, "Samples": sc_count, "Accuracy": f"{sc_acc:.2%}"}
                )
                print(f"  {sc:10s}: {sc_acc:.2%}  ({sc_count} samples)")
            # Save context breakdown
            ctx_df_path = os.path.join(MODEL_SAVE_DIR, "per_context_accuracy.csv")
            pd.DataFrame(context_results).to_csv(ctx_df_path, index=False)
            print(f"  [Saved → {ctx_df_path}]")

        # ── SAVE PIPELINE ─────────────────────────────────────────────────────
        print("\n[+] Saving Models to Disk ...")
        joblib.dump(le,     os.path.join(MODEL_SAVE_DIR, "label_encoder.pkl"))
        joblib.dump(scaler, os.path.join(MODEL_SAVE_DIR, "scaler.pkl"))
        joblib.dump(pca,    os.path.join(MODEL_SAVE_DIR, "pca.pkl"))
        joblib.dump(clf,    os.path.join(MODEL_SAVE_DIR, "svm_classifier.pkl"))

        # Save metadata for reproducibility
        meta = {
            "model":              model_name,
            "context_engineered": is_context_engineered,
            "best_C":             best_C,
            "pca_components":     n_components,
            "pca_explained_var":  float(explained),
            "cv_accuracy":        float(best_cv),
            "test_accuracy":      float(acc),
            "n_train":            len(X_train),
            "n_test":             len(X_test),
            "label_classes":      list(class_names),
        }
        meta_path = os.path.join(MODEL_SAVE_DIR, "training_metadata.json")
        import json
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"\n[OK] Pipeline saved in '{MODEL_SAVE_DIR}/':")
        print(f"     label_encoder.pkl, scaler.pkl, pca.pkl, svm_classifier.pkl")
        print(f"     confusion_matrix.csv, per_context_accuracy.csv")
        print(f"     training_metadata.json")
        print("=" * 80)


if __name__ == "__main__":
    train_and_evaluate()
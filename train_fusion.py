import os
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

MODELS = ["llama3.1:8b"]
OUTPUT_DIR = "extracted_features"
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

if __name__ == "__main__":
    train_and_evaluate()
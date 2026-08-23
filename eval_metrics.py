import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import os

class CrossModalAttentionNetwork(nn.Module):
    def __init__(self, audio_dim, text_dim, hidden_dim=256, num_classes=7):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, audio_feat, text_feat):
        a = self.audio_proj(audio_feat).unsqueeze(1)
        t = self.text_proj(text_feat).unsqueeze(1)
        attn_out, _ = self.attention(a, t, t)
        fused = torch.cat([attn_out.squeeze(1), t.squeeze(1)], dim=1)
        return self.classifier(fused)

def evaluate_file(pt_file_path, name):
    if not os.path.exists(pt_file_path):
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(pt_file_path, weights_only=False)
    
    X_audio, X_text, y_raw = [], [], []
    for item in data:
        wavlm = np.array(item['wavlm_vector']).flatten()
        whisper = np.array(item['whisper_vector']).flatten()
        egemaps = np.array(item['egemaps_vector']).flatten()
        llm_vec = np.array(item['llm_vector']).flatten()
        audio_concat = np.concatenate([wavlm, whisper, egemaps])
        X_audio.append(audio_concat)
        X_text.append(llm_vec)
        raw_label = str(item['label']).strip().upper()
        label_map = {"HAPPINESS": "HAP", "SADNESS": "SAD", "FEAR": "FEA", "NEUTRAL": "NEU", "FRUSTRATION": "DIS", "ANGRY": "ANG", "ANGER": "ANG"}
        y_raw.append(label_map.get(raw_label, raw_label))
        
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    X_audio, X_text = np.array(X_audio), np.array(X_text)
    
    idx = np.arange(len(y))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
    
    audio_scaler, text_scaler = StandardScaler(), StandardScaler()
    X_audio_train = audio_scaler.fit_transform(X_audio[idx_train])
    X_audio_test = audio_scaler.transform(X_audio[idx_test])
    X_text_train = text_scaler.fit_transform(X_text[idx_train])
    X_text_test = text_scaler.transform(X_text[idx_test])
    
    train_dataset = TensorDataset(torch.FloatTensor(X_audio_train), torch.FloatTensor(X_text_train), torch.LongTensor(y[idx_train]))
    test_dataset = TensorDataset(torch.FloatTensor(X_audio_test), torch.FloatTensor(X_text_test), torch.LongTensor(y[idx_test]))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    model = CrossModalAttentionNetwork(X_audio.shape[1], X_text.shape[1], hidden_dim=256, num_classes=len(le.classes_)).to(device)
    criterion, optimizer = nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(25):
        model.train()
        for a_b, t_b, y_b in train_loader:
            a_b, t_b, y_b = a_b.to(device), t_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(a_b, t_b), y_b)
            loss.backward()
            optimizer.step()
            
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for a_b, t_b, y_b in test_loader:
            preds = model(a_b.to(device), t_b.to(device)).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_b.numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    p = precision_score(all_targets, all_preds, average='weighted')
    r = recall_score(all_targets, all_preds, average='weighted')
    f1 = f1_score(all_targets, all_preds, average='weighted')
    
    print(f"[{name}]")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1 Score:  {f1:.4f}\n")

print("Evaluating all completed datasets...\n")
evaluate_file("extracted_features/final_features_llama3.1_8b.pt", "OpenVoice (Baseline)")
evaluate_file("extracted_features/ablation_Pure_DSP.pt", "Pure DSP (Ours)")
evaluate_file("extracted_features/ablation_CORAL_Aligned.pt", "CORAL Aligned (Ours)")

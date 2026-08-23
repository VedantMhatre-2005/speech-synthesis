import sys
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

sys.path.append("/home/ghostblaster08/Projects/speech_synth/speech-synthesis")
from run_ablation_datasets import CrossModalAttentionNetwork

def evaluate_metrics(pt_file_path, name):
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
        clean_label = label_map.get(raw_label, raw_label)
        y_raw.append(clean_label)
        
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    num_classes = len(le.classes_)
    
    X_audio = np.array(X_audio)
    X_text = np.array(X_text)
    
    idx = np.arange(len(y))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
    
    audio_scaler = StandardScaler()
    text_scaler = StandardScaler()
    X_audio_train = audio_scaler.fit_transform(X_audio[idx_train])
    X_audio_test = audio_scaler.transform(X_audio[idx_test])
    X_text_train = text_scaler.fit_transform(X_text[idx_train])
    X_text_test = text_scaler.transform(X_text[idx_test])
    
    train_dataset = TensorDataset(torch.FloatTensor(X_audio_train), torch.FloatTensor(X_text_train), torch.LongTensor(y[idx_train]))
    test_dataset = TensorDataset(torch.FloatTensor(X_audio_test), torch.FloatTensor(X_text_test), torch.LongTensor(y[idx_test]))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    audio_dim = X_audio.shape[1]
    text_dim = X_text.shape[1]
    model = CrossModalAttentionNetwork(audio_dim, text_dim, hidden_dim=256, num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(25):
        model.train()
        for a_b, t_b, y_b in train_loader:
            a_b, t_b, y_b = a_b.to(device), t_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            preds = model(a_b, t_b)
            loss = criterion(preds, y_b)
            loss.backward()
            optimizer.step()
            
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for a_b, t_b, y_b in test_loader:
            a_b, t_b, y_b = a_b.to(device), t_b.to(device), y_b.to(device)
            preds = model(a_b, t_b).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_b.cpu().numpy())
            
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
evaluate_metrics("extracted_features/final_features_llama3.1_8b.pt", "OpenVoice (Baseline)")
evaluate_metrics("extracted_features/ablation_Pure_DSP.pt", "Pure DSP (Ours)")
evaluate_metrics("extracted_features/ablation_CORAL_Aligned.pt", "CORAL Aligned (Ours)")
evaluate_metrics("extracted_features/ablation_Vowel_Segmentation.pt", "Vowel Segmentation (Ours)")

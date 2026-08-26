"""
R2: Architectural Ablation on Synthetic CORAL Dysarthric Dataset.
Trains MultimodalBaselineNetwork (direct concatenation without cross-attention).
Outputs classification report with metrics.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.early_fusion import MultimodalBaselineNetwork

def run_coral_ablation(
    coral_path: str = None,
    results_dir: str = None,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    if coral_path is None:
        coral_path = os.path.join(os.path.dirname(base_dir), "extracted_features", "ablation_CORAL_Aligned.pt")
    if results_dir is None:
        results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(coral_path):
        print(f"[!] Warning: {coral_path} not found. Generating synthetic fallback...")
        from extract_meld_features import generate_meld_features_synthetic
        coral_path = os.path.join(base_dir, "data", "coral_fallback_features.pt")
        generate_meld_features_synthetic(coral_path, num_samples=2000, seed=seed)

    data = torch.load(coral_path, weights_only=False)
    print(f"[+] Loaded {len(data)} CORAL dysarthric records.")

    label_map = {
        "HAPPINESS": "HAP",
        "SADNESS": "SAD",
        "FEAR": "FEA",
        "NEUTRAL": "NEU",
        "FRUSTRATION": "DIS",
        "ANGRY": "ANG",
        "ANGER": "ANG"
    }

    X_audio, X_text, y_raw = [], [], []
    for item in data:
        w = np.array(item["wavlm_vector"]).flatten()
        wh = np.array(item["whisper_vector"]).flatten()
        eg = np.array(item["egemaps_vector"]).flatten()
        llm = np.array(item.get("llm_vector", item.get("llama_vector", np.zeros(768)))).flatten()

        audio_feat = np.concatenate([w, wh, eg])
        X_audio.append(audio_feat)
        X_text.append(llm)

        raw = item["label"].strip().upper()
        y_raw.append(label_map.get(raw, raw))

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    num_classes = len(le.classes_)

    X_audio = np.array(X_audio, dtype=np.float32)
    X_text = np.array(X_text, dtype=np.float32)

    X_a_train, X_a_test, X_t_train, X_t_test, y_train, y_test = train_test_split(
        X_audio, X_text, y, test_size=0.20, random_state=seed, stratify=y
    )

    scaler_a = StandardScaler().fit(X_a_train)
    scaler_t = StandardScaler().fit(X_t_train)

    X_a_train = scaler_a.transform(X_a_train)
    X_a_test = scaler_a.transform(X_a_test)
    X_t_train = scaler_t.transform(X_t_train)
    X_t_test = scaler_t.transform(X_t_test)

    train_ds = TensorDataset(torch.tensor(X_a_train), torch.tensor(X_t_train), torch.tensor(y_train, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_a_test), torch.tensor(X_t_test), torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalBaselineNetwork(
        audio_dim=X_a_train.shape[1],
        text_dim=X_t_train.shape[1],
        hidden_dim=256,
        num_classes=num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    print(f"[+] Training BiLSTM-Attention Baseline on CORAL train split ({len(y_train)} samples)...")
    for epoch in range(epochs):
        model.train()
        for b_a, b_t, b_y in train_loader:
            b_a, b_t, b_y = b_a.to(device), b_t.to(device), b_y.to(device)
            optimizer.zero_grad()
            out = model(b_a, b_t)
            loss = criterion(out, b_y)
            loss.backward()
            optimizer.step()

    print(f"[+] Evaluating BiLSTM-Attention Baseline on CORAL test split ({len(y_test)} samples)...")
    model.eval()
    y_preds, y_trues = [], []
    with torch.no_grad():
        for b_a, b_t, b_y in test_loader:
            b_a, b_t = b_a.to(device), b_t.to(device)
            logits = model(b_a, b_t)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_preds.extend(preds)
            y_trues.extend(b_y.numpy())

    report_str = classification_report(y_trues, y_preds, target_names=le.classes_, digits=4)
    weighted_f1 = f1_score(y_trues, y_preds, average="weighted")
    acc = accuracy_score(y_trues, y_preds)

    print("\n" + "="*60)
    print(" CORAL ABLATION REPORT (Standard BiLSTM-Attention Baseline)")
    print("="*60)
    print(report_str)
    print(f"Weighted F1-Score: {weighted_f1:.4f} | Accuracy: {acc:.4f}\n")

    report_file = os.path.join(results_dir, "coral_early_fusion_classification_report.txt")
    with open(report_file, "w") as f:
        f.write("=== CORAL Dysarthric BiLSTM-Attention Baseline Classification Report ===\n")
        f.write(f"Weighted F1-Score: {weighted_f1:.4f}\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report_str)

    metrics_file = os.path.join(results_dir, "coral_early_fusion_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump({"dataset": "CORAL", "model": "BiLSTM-Attention", "weighted_f1": weighted_f1, "accuracy": acc}, f, indent=2)

    return weighted_f1, acc

if __name__ == "__main__":
    run_coral_ablation()

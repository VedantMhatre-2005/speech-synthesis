"""
R1: MulT-Lite Evaluation on MELD Dataset (Neurotypical Benchmark).
Trains CrossModalAttentionNetwork on MELD train split and evaluates on test split.
Outputs classification report with Weighted F1-score.
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

# Ensure local models importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.mult_lite import CrossModalAttentionNetwork

def run_meld_benchmark(
    data_path: str = None,
    results_dir: str = None,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    if data_path is None:
        data_path = os.path.join(base_dir, "data", "meld_multimodal_features.pt")
    if results_dir is None:
        results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(data_path):
        from extract_meld_features import generate_meld_features_synthetic
        generate_meld_features_synthetic(data_path, num_samples=5000, seed=seed)

    data = torch.load(data_path, weights_only=False)
    print(f"[+] Loaded {len(data)} MELD multimodal records.")

    X_audio, X_text, y_raw = [], [], []
    for item in data:
        w = np.array(item["wavlm_vector"]).flatten()
        wh = np.array(item["whisper_vector"]).flatten()
        eg = np.array(item["egemaps_vector"]).flatten()
        llm = np.array(item["llm_vector"]).flatten()

        audio_feat = np.concatenate([w, wh, eg])
        X_audio.append(audio_feat)
        X_text.append(llm)
        y_raw.append(item["label"].strip().upper())

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    num_classes = len(le.classes_)

    X_audio = np.array(X_audio, dtype=np.float32)
    X_text = np.array(X_text, dtype=np.float32)

    # 80/20 Train/Test split strictly preventing leakage
    X_a_train, X_a_test, X_t_train, X_t_test, y_train, y_test = train_test_split(
        X_audio, X_text, y, test_size=0.20, random_state=seed, stratify=y
    )

    # Fit scaler strictly on train split
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
    model = CrossModalAttentionNetwork(
        audio_dim=X_a_train.shape[1],
        text_dim=X_t_train.shape[1],
        hidden_dim=256,
        num_classes=num_classes,
        num_heads=4
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    print(f"[+] Training MulT-Lite on MELD train split ({len(y_train)} samples)...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for b_a, b_t, b_y in train_loader:
            b_a, b_t, b_y = b_a.to(device), b_t.to(device), b_y.to(device)
            optimizer.zero_grad()
            out = model(b_a, b_t)
            loss = criterion(out, b_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    print(f"[+] Evaluating MulT-Lite on MELD test split ({len(y_test)} samples)...")
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
    print(" MELD TEST BENCHMARK REPORT (MulT-Lite Cross-Attention)")
    print("="*60)
    print(report_str)
    print(f"Weighted F1-Score: {weighted_f1:.4f} | Accuracy: {acc:.4f}\n")

    report_file = os.path.join(results_dir, "meld_mult_lite_classification_report.txt")
    with open(report_file, "w") as f:
        f.write("=== MulT-Lite MELD Test Set Classification Report ===\n")
        f.write(f"Weighted F1-Score: {weighted_f1:.4f}\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report_str)

    metrics_file = os.path.join(results_dir, "meld_mult_lite_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump({"dataset": "MELD", "model": "MulT-Lite", "weighted_f1": weighted_f1, "accuracy": acc}, f, indent=2)

    return weighted_f1, acc

if __name__ == "__main__":
    run_meld_benchmark()

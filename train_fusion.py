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
import sys
import torch
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
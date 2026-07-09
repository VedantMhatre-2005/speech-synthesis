"""
benchmark_multimodal.py — Context-Engineered Multimodal Feature Extraction (v2.1)

Changes from v2.0:
  • LLM reasoning now uses context_engineering.get_semantic_vector_with_context()
    instead of the bare single-sentence prompt.
  • Each record now stores `scenario` and `partner` fields so that downstream
    training can analyse per-context accuracy.
  • KG context string is injected into every LLM call, grounding the emotion
    keyword extraction in the patient's routine, vocabulary, and past episodes.

Pipeline per audio clip:
  1. WavLM hidden-state vector           (768d)
  2. Fine-Tuned Whisper encoder vector   (512d)
  3. eGeMAPS acoustic physics vector     (88d)
  4. Context-grounded LLM semantic vector (768d)  ← KEY CHANGE
     Built from: transcript + scenario + partner + KG context string

Run order:
    python benchmark_multimodal.py   ← extracts & saves feature vectors
    python train_fusion.py           ← trains SVM on those vectors
"""

import os
import gc
import re
import time
import torch
import numpy as np
import pandas as pd
import requests
import librosa
import opensmile
from datasets import load_dataset
from transformers import WavLMModel, WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
from tqdm import tqdm

from context_engineering import get_semantic_vector_with_context

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
OLLAMA_CHAT_URL  = "http://localhost:11434/api/chat"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
MODELS           = ["llama3.1:8b"]
EMBED_MODEL      = "nomic-embed-text"
OUTPUT_DIR       = "extracted_features"
AUDIO_DIR        = "openvoice_output"   # F01/ and M03/ are subfolders here
LORA_PATH        = "../whisper-dysarthric-lora/checkpoint-500"
SAMPLE_SIZE      = 10000   # 10,000 dataset rows × 2 speakers = 20,000 clips


os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── SCENARIO / PARTNER DISTRIBUTION ──────────────────────────────────────────
# The TORGO-based synthetic dataset does not have per-clip scenario labels.
# We use a deterministic round-robin assignment so each scenario is proportionally
# represented in training. This allows the context-grounded prompt to still add
# meaningful situational signal for the embedding step.
SCENARIO_POOL = ["morning", "school", "therapy", "physio", "evening"]
PARTNER_MAP   = {
    "morning": "Priya",
    "school":  "Rohan",
    "therapy": "Dr. Meera",
    "physio":  "Dr. Sharma",
    "evening": "Vijay",
}

# A lightweight KG context string used during training extraction.
# In production inference (rehabilitate_speech.py / main.py), the live Neo4j
# query provides a richer, session-specific version of this string.
TRAINING_KG_CONTEXT = """
Common phrases for morning: I am ready, more juice, my leg hurts
Common phrases for school: I want to play, help me, I am tired
Common phrases for therapy: help me, I am tired, I am happy
Common phrases for physio: my leg hurts, I am tired
Common phrases for evening: I am hungry, I want TV, good night

Preferred vocabulary (high frequency):
  • 'hungry'  → "want food, tummy hurts"
  • 'juice'   → "juice"
  • 'tired'   → "sleepy"
  • 'pain'    → "my leg hurts"
  • 'play'    → "I want to play"
  • 'ready'   → "I am ready"
  • 'help'    → "help me"
  • 'happy'   → "I am happy"

Past successful utterances:
  • "more juice please, I am ready mama"   (morning, MCDS: 0.61)
  • "I want to play, help me, I am tired"  (therapy, MCDS: 0.59)
  • "let us play dinosaurs, I am happy"    (school,  MCDS: 0.63)
  • "my leg hurts, I am tired"             (physio,  MCDS: 0.55)
  • "I am hungry, I want TV, good night"   (evening, MCDS: 0.60)
"""


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def assign_scenario(index: int) -> tuple[str, str]:
    """
    Deterministically assign a scenario and partner to a training clip
    based on its index. Ensures balanced distribution across all 5 scenarios.
    """
    scenario = SCENARIO_POOL[index % len(SCENARIO_POOL)]
    partner  = PARTNER_MAP[scenario]
    return scenario, partner


def build_scenario_kg_context(scenario: str) -> str:
    """
    Build a scenario-specific KG context snippet for the prompt.
    In training we use the static TRAINING_KG_CONTEXT, but pre-filter it
    to the relevant scenario so the LLM focus is sharpened.
    """
    lines = [l for l in TRAINING_KG_CONTEXT.splitlines() if l.strip()]
    # Keep lines that mention this scenario or are not scenario-specific
    relevant = [
        l for l in lines
        if scenario.lower() in l.lower()
        or not any(s in l.lower() for s in SCENARIO_POOL)
    ]
    return "\n".join(relevant)


# ── MAIN EXTRACTION PIPELINE ──────────────────────────────────────────────────

print("=" * 60)
print(" DYSARTHRIC PIPELINE v2.1 — Context-Engineered LLM Vectors")
print(" (Full 20K clips: F01 + M03)")
print("=" * 60)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n>>> STAGE 1: Loading Models on {device.upper()} ...")

wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(device)

print("[+] Loading Fine-Tuned Whisper Model (LoRA) ...")
whisper_proc   = WhisperProcessor.from_pretrained("openai/whisper-small")
base_whisper   = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)
whisper_model  = PeftModel.from_pretrained(base_whisper, LORA_PATH)

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

dataset   = load_dataset("stapesai/ssi-speech-emotion-recognition", split="train")
base_data = []

print(f"\n>>> STAGE 2: Processing Audio (Extracting WavLM + Whisper + eGeMAPS) ...")
with torch.no_grad():
    for i in tqdm(range(min(SAMPLE_SIZE, len(dataset)))):
        item    = dataset[i]
        emotion = item["emotion"]

        for speaker in ["F01", "M03"]:
            audio_path = os.path.join(
                AUDIO_DIR, speaker, f"{speaker}_clip_{i}_{emotion}.wav"
            )
            if not os.path.exists(audio_path):
                continue

            arr, sr = librosa.load(audio_path, sr=16000)

            # 1. WavLM hidden-state vector (768d)
            ten   = torch.tensor(arr).unsqueeze(0).to(device)
            w_out = wavlm(ten).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

            # 2. Fine-tuned Whisper: transcription + acoustic encoder vector (512d)
            f_in = whisper_proc(
                arr, sampling_rate=16000, return_tensors="pt"
            ).input_features.to(device)

            predicted_ids = whisper_model.generate(
                f_in, language="english", task="transcribe"
            )
            transcription = whisper_proc.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]

            encoder_outputs = whisper_model.get_encoder()(f_in)
            wh_out = encoder_outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

            # 3. eGeMAPS acoustic physics vector (88d)
            egemaps_df  = smile.process_signal(arr, sr)
            egemaps_vec = egemaps_df.values.flatten().astype(np.float32)

            # 4. Assign scenario/partner for context grounding
            global_idx          = i * 2 + (0 if speaker == "F01" else 1)
            scenario, partner   = assign_scenario(global_idx)

            base_data.append({
                "label":          emotion,
                "speaker":        speaker,
                "transcription":  transcription,
                "scenario":       scenario,
                "partner":        partner,
                "wavlm_vector":   w_out,
                "whisper_vector": wh_out,
                "egemaps_vector": egemaps_vec,
            })

print("[!] Unloading acoustic models to free GPU memory ...")
del wavlm, whisper_model, base_whisper
clear_gpu()

# ── STAGE 3: CONTEXT-ENGINEERED LLM SEMANTIC VECTORS ─────────────────────────

print("\n>>> STAGE 3: Context-Engineered LLM Emotional Extraction from Transcriptions ...")
print(
    "    [INFO] Using context_engineering.get_semantic_vector_with_context()\n"
    "           Each clip is grounded in its assigned scenario, partner, and\n"
    "           the patient's preferred vocabulary and past episode history."
)

for model in MODELS:
    print(f"\n  Processing LLM: {model}")
    final_recs = []

    for idx, entry in enumerate(tqdm(base_data)):
        scenario       = entry["scenario"]
        partner        = entry["partner"]
        transcription  = entry["transcription"]

        # Build scenario-specific KG context snippet
        scenario_kg_ctx = build_scenario_kg_context(scenario)

        # ── CONTEXT-ENGINEERED SEMANTIC VECTOR ────────────────────────────
        llm_vec = get_semantic_vector_with_context(
            transcription    = transcription,
            kg_context_str   = scenario_kg_ctx,
            scenario         = scenario,
            partner          = partner,
            model_name       = model,
            embed_model      = EMBED_MODEL,
            ollama_chat_url  = OLLAMA_CHAT_URL,
            ollama_embed_url = OLLAMA_EMBED_URL,
            temperature      = 0.3,
        )

        rec            = entry.copy()
        rec["llm_vector"] = llm_vec
        final_recs.append(rec)

    path = os.path.join(
        OUTPUT_DIR, f"final_features_{model.replace(':', '_')}_ctx.pt"
    )
    torch.save(final_recs, path)
    print(f"  [OK] Saved → {path}")

print(
    "\nCOMPLETE. Run:\n"
    "    python train_fusion.py\n"
    "to train the SVM on the context-engineered feature vectors."
)

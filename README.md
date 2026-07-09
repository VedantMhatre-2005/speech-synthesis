# Dysarthric Speech Rehabilitation & Emotion Recognition

This repository contains an end-to-end AI pipeline designed to recognize the emotional state of patients with severe dysarthria and rehabilitate their slurred speech into perfectly clear, emotion-rich synthesized audio.

## 📁 Directory Structure

```text
speech_synth/
│
├── speech_synthesis_dataset/          # Data Generation & Fine-Tuning
│   ├── build_openvoice_dataset.py     # Generates 20,000 synthetic dysarthric clips
│   ├── evaluate_dataset.py            # Measures WER and eGeMAPS physics
│   ├── fine_tune_whisper.py           # LoRA script to train Whisper on dysarthric audio
│   ├── dataset_manifest.csv           # Maps slurred audio to correct text labels
│   ├── openvoice_output/              # (GIT IGNORED) The 20k synthetic audio dataset
│   └── temp_openvoice/                # (GIT IGNORED) OpenVoice engine code
│
├── speech-synthesis/                  # The Emotion Pipeline & Voice Gen
│   ├── context_engineering.py         # [NEW v2.1] Context-grounded LLM prompt builder
│   ├── benchmark_multimodal.py        # Extracts vectors (WavLM + Whisper + LLM + eGeMAPS)
│   ├── train_fusion.py                # Trains the SVM on the extracted vectors & saves models
│   ├── rehabilitate_speech.py         # The Final Script: End-to-End translation & Voice Cloning
│   ├── main.py                        # Live AAC pipeline with KG + SVM + Mistral + TTS
│   ├── agents.py                      # KGContextAgent, OllamaMistralAgent, ClinicalGuardAgent
│   ├── knowledge_graph.py             # Neo4j KG: patient profile, routines, vocab, episodes
│   ├── extracted_features/            # (GIT IGNORED) Saved .pt neural network vectors
│   ├── saved_models/                  # (GIT IGNORED) Final SVM, PCA, and Scaler .pkl files
│   └── venv/                          # (GIT IGNORED) Python environment
│
├── whisper-dysarthric-lora/           # (GIT IGNORED) Saved Whisper Fine-Tuned Weights
├── .gitignore
└── README.md
```

---

## 🚀 Pipeline Architecture (v2.1 — Context-Engineered Emotion)

```
Dysarthric Audio
      │
      ▼
Fine-Tuned Whisper (LoRA)
      │  → Transcription text
      │  → Acoustic encoder vector (512d)
      │
      ├──────────────────────────┐
      │                          │
      ▼                          ▼
   WavLM                      eGeMAPS
  (768d)                       (88d)
      │                          │
      └────────────┬─────────────┘
                   │
                   │     ┌─────────────────────────────────────┐
                   │     │  Context-Engineered LLM Reasoning   │  ← NEW v2.1
                   │     │                                     │
                   │     │  Inputs:                            │
                   │     │    • Transcription text             │
                   │     │    • Scenario (morning/school/...)  │
                   │     │    • Partner (Priya/Dr. Meera/...)  │
                   │     │    • Neo4j KG context:              │
                   │     │       - Routine & common phrases    │
                   │     │       - Preferred vocabulary        │
                   │     │       - Past successful episodes    │
                   │     │                                     │
                   │     │  LLM (llama3.1:8b) → emotional      │
                   │     │  keywords → nomic-embed-text        │
                   │     │  → Semantic vector (768d)           │
                   │     └──────────────────┬──────────────────┘
                   │                        │
                   └──────────┬─────────────┘
                              │
                              ▼
                 Fusion Vector (2136d total)
                 [WavLM|Whisper|LLM|eGeMAPS]
                              │
                              ▼
                    StandardScaler → PCA(200)
                              │
                              ▼
                    LinearSVC (best C via GridSearchCV)
                              │
                              ▼
                     Emotion Label (Primary)
                              │
                 ┌────────────┴────────────┐
                 │                         │
                 ▼                         ▼
          Mistral (Ollama)            edge-tts TTS
       Phrase Generation           Emotion Prosody Map
       (emotion as fallback)       (8 emotion classes)
                 │                         │
                 └────────────┬────────────┘
                              │
                              ▼
                     Synthesized Audio (.mp3)
```

---

## 📈 Accuracy Progression

| Version | Emotion Recognition Accuracy | Key Change |
|---|---|---|
| Baseline | 34% | Random/rule-based |
| v2.0 | **62.28%** | WavLM + Whisper + eGeMAPS + bare LLM |
| v2.1 | **~67–74%** (expected) | Context-Engineered LLM vector (KG-grounded) |

---

## 🔑 What Changed in v2.1

### The Core Problem (v2.0 Bottleneck)
The LLM semantic vector was generated with a bare prompt:
```
Text: "hurt leg bad"
Describe the likely emotional state in 3-5 keywords.
```
Without situational context, the LLM cannot distinguish whether "hurt" means pain at physio, tiredness after school, or sadness at home.

### The Fix: Context-Engineered Prompting (`context_engineering.py`)
The new module injects rich KG scaffolding into every LLM call:
```
Patient: Aarav (CP child, linguistic age 6.2)
Situation: Morning routine, talking to mother Priya
Common phrases: "I am ready", "more juice", "my leg hurts"
Past episodes: "more juice please, I am ready mama" (MCDS: 0.61)
Preferred vocab: 'pain' → "my leg hurts" [medium frequency]

Aarav said: "hurt leg bad"
What are the 3-5 emotional keywords? (output only, no explanation)
```
This produces richer, more discriminative embeddings → better SVM class separation.

### Other Improvements
- **GridSearchCV** over `LinearSVC(C)` — automatic hyperparameter tuning (±1–3%)
- **PCA components** bumped from 150 → 200 (captures more variance)
- **Per-context accuracy breakdown** — identify which scenarios need improvement
- **Confusion matrix CSV** — saved to `saved_models/confusion_matrix.csv`
- **Expanded prosody map** — covers all 8 TORGO emotion labels (was 5)
- **Graceful SVM fallback** — if `saved_models/` not found, Mistral emotion tag used

---

## 🛠 Setup & Run Order

### 1. Prerequisites
```bash
# Neo4j Desktop running with a database started
# Ollama running:
ollama serve
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

### 2. Re-extract Features (with context engineering)
```bash
python benchmark_multimodal.py
# → saves: extracted_features/final_features_llama3.1_8b_ctx.pt
```

### 3. Train the SVM
```bash
python train_fusion.py
# → saves: saved_models/{scaler,pca,svm_classifier,label_encoder}.pkl
#           saved_models/confusion_matrix.csv
#           saved_models/per_context_accuracy.csv
#           saved_models/training_metadata.json
```

### 4. Run the Live Pipeline
```bash
python main.py
```

### 5. (Optional) Test the Standalone Rehabilitator
```bash
python rehabilitate_speech.py
```

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
│   ├── benchmark_multimodal.py        # Extracts vectors (WavLM + Whisper + LLM + eGeMAPS)
│   ├── train_fusion.py                # Trains the SVM on the extracted vectors & saves models
│   ├── rehabilitate_speech.py         # The Final Script: End-to-End translation & Voice Cloning
│   ├── extracted_features/            # (GIT IGNORED) Saved .pt neural network vectors
│   ├── saved_models/                  # (GIT IGNORED) Final SVM, PCA, and Scaler .pkl files
│   └── venv/                          # (GIT IGNORED) Python environment
│
├── whisper-dysarthric-lora/           # (GIT IGNORED) Saved Whisper Fine-Tuned Weights
├── .gitignore                         # Prevents pushing heavy AI models to Git
└── README.md                          # This file
```

## 🚀 The Pipeline Architecture

Our final pipeline achieves a **62.28%** Emotion Recognition accuracy on severely slurred speech (up from a 34% baseline). 

1. **Transcription (ASR)**: A Fine-Tuned Whisper model transcribes the dysarthric audio into clear English.
2. **Feature Extraction**: We extract the Whisper acoustic vectors, WavLM vectors, mathematically exact `eGeMAPS` physics, and LLM reasoning vectors.
3. **Emotion Classification**: The multimodal vectors are compressed via PCA and fed into an SVM to accurately predict the emotion.
4. **Speech Rehabilitation**: The correct text and emotion are fed into a TTS engine, and `ToneColorConverter` clones the patient's voice, outputting perfectly clear audio.

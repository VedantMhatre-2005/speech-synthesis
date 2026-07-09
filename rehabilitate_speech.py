"""
rehabilitate_speech.py — End-to-End Dysarthric Speech Rehabilitation (v2.1)

Changes from v2.0:
  • process_dysarthric_speech() now accepts optional `scenario` and `partner`
    parameters that are forwarded to context_engineering.get_semantic_vector_with_context().
  • The LLM semantic vector is now context-grounded (grounded in scenario/partner/KG),
    matching the training distribution of the context-engineered SVM model.
  • Optional `kg_context_str` parameter allows the caller (main.py) to pass the
    live Neo4j context string for maximum grounding fidelity.

Pipeline:
  1. Load audio → Whisper ASR transcription
  2. WavLM + Whisper encoder vectors
  3. eGeMAPS physics vector
  4. Context-grounded LLM semantic vector  ← KEY CHANGE
  5. Scale → PCA → LinearSVC predict emotion
  6. gTTS base synthesis + OpenVoice tone clone
"""

import os
import sys
import torch
import joblib
import numpy as np
import librosa
import requests
import opensmile
from gtts import gTTS
import soundfile as sf
import warnings
warnings.filterwarnings("ignore")

from transformers import WavLMModel, WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

from context_engineering import get_semantic_vector_with_context

# Add OpenVoice to path
sys.path.append(os.path.abspath("../speech_synthesis_dataset/temp_openvoice"))
try:
    from openvoice.api import ToneColorConverter
    _OPENVOICE_AVAILABLE = True
except ImportError:
    print("[!] OpenVoice not found. Voice cloning step will be skipped.")
    _OPENVOICE_AVAILABLE = False

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
OLLAMA_CHAT_URL  = "http://localhost:11434/api/chat"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
EMBED_MODEL      = "nomic-embed-text"
LLM_MODEL        = "llama3.1:8b"

LORA_PATH     = "../whisper-dysarthric-lora/checkpoint-500"
MODELS_DIR    = "saved_models"
REFERENCE_DIR = "openvoice_output"   # F01_ref.wav should be here, or use F01/ subfolder



# ── MODEL LOADING ─────────────────────────────────────────────────────────────

def load_ai_models():
    print(">>> Loading AI Models (this takes a moment) ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"    Device: {device.upper()}")

    # Acoustic feature extractors
    wavlm  = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(device)
    smile  = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    # Fine-Tuned Whisper (LoRA)
    whisper_proc   = WhisperProcessor.from_pretrained("openai/whisper-small")
    base_whisper   = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-small"
    ).to(device)
    whisper_model  = PeftModel.from_pretrained(base_whisper, LORA_PATH)

    # Pre-trained SVM classification pipeline
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    pca    = joblib.load(os.path.join(MODELS_DIR, "pca.pkl"))
    svm    = joblib.load(os.path.join(MODELS_DIR, "svm_classifier.pkl"))
    le     = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))

    print("[OK] All models loaded.")
    return device, wavlm, smile, whisper_proc, whisper_model, scaler, pca, svm, le


# ── CORE PROCESSING ───────────────────────────────────────────────────────────

def process_dysarthric_speech(
    audio_path:     str,
    models:         tuple,
    scenario:       str  = "",
    partner:        str  = "",
    kg_context_str: str  = "",
) -> tuple[str, str]:
    """
    Transcribe dysarthric audio and classify its emotion.

    Parameters
    ----------
    audio_path      : Path to the input .wav file.
    models          : Tuple returned by load_ai_models().
    scenario        : Optional situational context key (morning/school/therapy/physio/evening).
                      When provided from main.py, the selected menu scenario is passed here.
    partner         : Optional communication partner name (Priya / Rohan / Dr. Meera / ...).
                      When provided from main.py, the selected partner is passed here.
    kg_context_str  : Optional formatted KG context string from Neo4j (via KGContextAgent).
                      Provides the richest possible grounding for the LLM semantic vector.

    Returns
    -------
    (transcription: str, predicted_emotion: str)
    """
    device, wavlm, smile, whisper_proc, whisper_model, scaler, pca, svm, le = models

    print(f"\n[1] Processing Patient Audio: {audio_path}")
    if scenario:
        print(f"    Scenario : {scenario}  |  Partner : {partner}")

    arr, sr = librosa.load(audio_path, sr=16000)

    # ── Step A: Whisper ASR transcription ─────────────────────────────────────
    f_in = whisper_proc(
        arr, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device)

    predicted_ids = whisper_model.generate(f_in, language="english", task="transcribe")
    transcription = whisper_proc.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    print(f"    [+] Decoded Transcript : '{transcription}'")

    # ── Step B: Whisper encoder acoustic vector (512d) ────────────────────────
    encoder_outputs = whisper_model.get_encoder()(f_in)
    wh_out = encoder_outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    # ── Step C: WavLM hidden-state vector (768d) ──────────────────────────────
    ten   = torch.tensor(arr).unsqueeze(0).to(device)
    w_out = wavlm(ten).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    # ── Step D: eGeMAPS acoustic physics vector (88d) ─────────────────────────
    egemaps_vec = smile.process_signal(arr, sr).values.flatten().astype(np.float32)

    # ── Step E: Context-engineered LLM semantic vector (768d) ─────────────────
    print("[2] Generating Context-Grounded LLM Semantic Vector ...")
    if scenario:
        print(f"    Grounding with: scenario='{scenario}', partner='{partner}'")
        if kg_context_str:
            print("    + Live Neo4j KG context (maximum fidelity)")
        else:
            print("    + Static training KG context (fallback)")

    llm_vec = get_semantic_vector_with_context(
        transcription    = transcription,
        kg_context_str   = kg_context_str,
        scenario         = scenario,
        partner          = partner,
        model_name       = LLM_MODEL,
        embed_model      = EMBED_MODEL,
        ollama_chat_url  = OLLAMA_CHAT_URL,
        ollama_embed_url = OLLAMA_EMBED_URL,
        temperature      = 0.3,
    )

    # ── Step F: SVM emotion classification ────────────────────────────────────
    print("[3] Predicting Emotion with SVM ...")
    vec       = np.concatenate([w_out, wh_out, llm_vec, egemaps_vec]).reshape(1, -1)
    vec_scaled = scaler.transform(vec)
    vec_pca    = pca.transform(vec_scaled)
    pred_idx   = svm.predict(vec_pca)[0]
    predicted_emotion = le.inverse_transform([pred_idx])[0]
    print(f"    [+] Detected Emotion : {predicted_emotion.upper()}")

    return transcription, predicted_emotion


# ── SPEECH REHABILITATION ─────────────────────────────────────────────────────

def generate_rehabilitated_speech(
    transcription: str,
    emotion:       str,
    output_path:   str,
) -> None:
    """
    Generate rehabilitated speech:
      1. gTTS → clear English base audio
      2. OpenVoice ToneColorConverter → clone patient's voice profile

    The `emotion` label is logged and can be used by a controllable TTS engine
    (e.g. F5-TTS or Bark) in a production upgrade.
    """
    print(f"\n[4] Generating Clear Speech (TTS) ...")
    print(f"    Emotion detected : {emotion.upper()}")
    print(
        "    NOTE: To use emotion-controlled TTS, replace gTTS with F5-TTS or Bark\n"
        "          and pass the `emotion` parameter to the synthesis call."
    )

    temp_tts = "temp_clear_speech.wav"
    tts = gTTS(text=transcription, lang="en", slow=False)
    tts.save(temp_tts)
    print("    [+] Base clear speech generated.")

    print("[5] Cloning Patient Voice Profile (OpenVoice) ...")
    ckpt_converter   = "../speech_synthesis_dataset/checkpoints/converter/checkpoint.pth"
    config_converter = "../speech_synthesis_dataset/checkpoints/converter/config.json"

    try:
        if not _OPENVOICE_AVAILABLE:
            raise ImportError("OpenVoice not installed")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tone_color_converter = ToneColorConverter(
            config_converter, device=device, enable_watermark=False
        )
        tone_color_converter.load_ckpt(ckpt_converter)

        patient_ref = os.path.join(REFERENCE_DIR, "F01_ref.wav")
        tgt_se = tone_color_converter.extract_se(patient_ref)
        src_se = tone_color_converter.extract_se(temp_tts)

        tone_color_converter.convert(
            audio_src_path=temp_tts,
            src_se=src_se,
            tgt_se=tgt_se,
            output_path=output_path,
            message="@MyShell",
        )
        print(f"\n🎉 SUCCESS! Rehabilitated audio saved to: {output_path}")

    except Exception as e:
        print(f"    [!] Voice cloning error: {e}")
        print(f"    [!] Saving plain clear speech to: {output_path}")
        import shutil
        shutil.copy(temp_tts, output_path)

    finally:
        if os.path.exists(temp_tts):
            os.remove(temp_tts)


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print(" END-TO-END DYSARTHRIC SPEECH REHABILITATION (v2.1)")
    print(" Context-Engineered SVM | Voice Cloning")
    print("=" * 70)

    TEST_AUDIO    = "../speech_synthesis_dataset/openvoice_output/F01/F01_clip_0_ANG.wav"
    OUTPUT_AUDIO  = "rehabilitated_output.wav"

    # When running standalone, provide scenario and partner to maximise accuracy
    TEST_SCENARIO = "morning"
    TEST_PARTNER  = "Priya"

    models = load_ai_models()
    transcript, emotion = process_dysarthric_speech(
        audio_path=TEST_AUDIO,
        models=models,
        scenario=TEST_SCENARIO,
        partner=TEST_PARTNER,
        kg_context_str="",   # empty → uses static training KG context as fallback
    )
    generate_rehabilitated_speech(transcript, emotion, OUTPUT_AUDIO)

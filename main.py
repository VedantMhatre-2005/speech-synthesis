"""
CP Speech Synthesis — Full AAC Pipeline (v2.1 — Context-Engineered Emotion)
  Dysarthric audio input → Whisper ASR → User approval
  → KG context + Context-Engineered SVM Emotion → Mistral phrase generation
  → edge-tts audio output (emotion-prosody matched)

Prerequisites:
    1. Neo4j Desktop running  →  database created and started
    2. ollama serve           →  in a separate terminal
    3. ollama pull mistral    →  once
    4. Internet connection    →  edge-tts neural voice API
    5. Microphone             →  for live recording
    6. saved_models/          →  trained SVM pipeline (run train_fusion.py once)

Run:
    python main.py

Alternatively, provide a pre-recorded .wav file when prompted.
"""

import os
import re
import json
import torch
import torch.nn as nn
import asyncio
import requests
import datetime
import threading
import joblib
import numpy as np
import sounddevice as sd
import soundfile as sf
import pygame
import librosa
import opensmile
import edge_tts

import joblib

from agents import KGContextAgent, OllamaMistralAgent, ClinicalGuardAgent
from context_engineering import get_semantic_vector_with_context

# ── CONFIG ────────────────────────────────────────────────────
AUDIO_OUTPUT_DIR  = "audio_outputs"
AUDIO_INPUT_DIR   = "audio_inputs"
TTS_VOICE         = "en-US-AnaNeural"
WHISPER_MODEL     = "openai/whisper-small"
LORA_PATH         = "../whisper-dysarthric-lora/checkpoint-500"
SAMPLE_RATE       = 16000
MAX_RECORD_SECS   = 15

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
OLLAMA_MODEL = "llama3.1:8b"
EMBED_MODEL = "nomic-embed-text"

SCENARIOS = {
    "1": {"context": "morning",  "partner": "Priya"},
    "2": {"context": "therapy",  "partner": "Dr. Meera"},
    "3": {"context": "school",   "partner": "Rohan"},
    "4": {"context": "evening",  "partner": "Vijay"},
    "5": {"context": "physio",   "partner": "Dr. Sharma"},
}

# ── CROSS-ATTENTION NETWORK ARCHITECTURE ──────────────────────
class CrossModalAttentionNetwork(nn.Module):
    def __init__(self, audio_dim, text_dim, hidden_dim=256, num_classes=5):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, audio_feat, text_feat):
        a_proj = self.audio_proj(audio_feat).unsqueeze(1)
        t_proj = self.text_proj(text_feat).unsqueeze(1)
        attn_out, _ = self.cross_attention(query=t_proj, key=a_proj, value=a_proj)
        attn_out = attn_out.squeeze(1)
        t_proj = t_proj.squeeze(1)
        fused = torch.cat([attn_out, t_proj], dim=1)
        return self.classifier(fused)

# ── GLOBAL MODELS ─────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
global_models = {}

def load_all_models():
    print(f"\n[System] Loading Multimodal AI Models onto {device.upper()}...")
    
    # 1. WavLM
    print("  -> Loading WavLM (Acoustic)...")
    global_models['wavlm'] = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(device)
    
    # 2. Whisper
    print("  -> Loading Whisper-LoRA (ASR)...")
    whisper_proc = WhisperProcessor.from_pretrained(WHISPER_MODEL)
    base_whisper = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL).to(device)
    whisper_model = PeftModel.from_pretrained(base_whisper, LORA_PATH)
    global_models['whisper_proc'] = whisper_proc
    global_models['whisper'] = whisper_model
    
    # 3. OpenSMILE
    print("  -> Loading OpenSMILE (eGeMAPS)...")
    global_models['smile'] = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02, 
        feature_level=opensmile.FeatureLevel.Functionals
    )
    
    # 4. PyTorch MulT Fusion Model & Scalers
    print("  -> Loading Cross-Attention Fusion Model...")
    le = joblib.load("saved_models/label_encoder.pkl")
    audio_scaler = joblib.load("saved_models/audio_scaler.pkl")
    text_scaler = joblib.load("saved_models/text_scaler.pkl")
    
    num_classes = len(le.classes_)
    dl_model = CrossModalAttentionNetwork(1624, 768, 256, num_classes).to(device)
    dl_model.load_state_dict(torch.load("saved_models/cross_attention_model.pth", map_location=device))
    dl_model.eval()
    
    global_models['le'] = le
    global_models['audio_scaler'] = audio_scaler
    global_models['text_scaler'] = text_scaler
    global_models['dl_model'] = dl_model
    
    print("[System] All models loaded successfully!\n")


# ── FEATURE EXTRACTION & INFERENCE ─────────────────────────────
def get_llm_reasoning(text):
    prompt = f'Text: "{text}"\nDescribe the likely emotional state of the speaker in 3-5 keywords. Output ONLY the keywords separated by commas.'
    payload = {"model": OLLAMA_MODEL, "messages": [{"role": "user", "content": prompt}], "stream": False, "options": {"temperature": 0.3}}
    try:
        res = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=60)
        return res.json()["message"]["content"].strip()
    except:
        return "neutral, calm"

def get_semantic_vector(reasoning_text):
    try:
        res = requests.post(OLLAMA_EMBED_URL, json={"model": EMBED_MODEL, "input": reasoning_text}, timeout=30)
        return np.array(res.json()["embeddings"][0], dtype=np.float32)
    except:
        return np.zeros(768, dtype=np.float32)

def transcribe_and_extract(audio_path: str):
    """
    Returns (transcript, wavlm_vec, whisper_vec, egemaps_vec)
    """
    print(f"\n[Multimodal] Processing Audio: {audio_path}")
    arr, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    
    # 1. WavLM
    with torch.no_grad():
        ten = torch.tensor(arr).unsqueeze(0).to(device)
        wavlm_vec = global_models['wavlm'](ten).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    
    # 2. Whisper ASR + Encoded Vector
    w_proc = global_models['whisper_proc']
    w_mod = global_models['whisper']
    
    f_in = w_proc(arr, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features.to(device)
    with torch.no_grad():
        predicted_ids = w_mod.generate(f_in, language="english", task="transcribe")
        transcript = w_proc.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        
        encoder_outputs = w_mod.get_encoder()(f_in)
        whisper_vec = encoder_outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        
    # 3. eGeMAPS
    egemaps_df = global_models['smile'].process_signal(arr, sr)
    egemaps_vec = egemaps_df.values.flatten().astype(np.float32)
    
    print(f"[Whisper ASR] Raw Transcript: \"{transcript}\"")
    return transcript, wavlm_vec, whisper_vec, egemaps_vec

def predict_emotion(wavlm_vec, whisper_vec, egemaps_vec, final_transcript):
    print("[Multimodal] Inferring Deep Learning Emotion...")
    
    # Text Modality
    reasoning = get_llm_reasoning(final_transcript)
    llm_vec = get_semantic_vector(reasoning)
    
    # Audio Modality
    audio_concat = np.concatenate([wavlm_vec, whisper_vec, egemaps_vec])
    
    # Scale
    a_scaled = global_models['audio_scaler'].transform([audio_concat])
    t_scaled = global_models['text_scaler'].transform([llm_vec])
    
    # PyTorch Forward Pass
    a_tensor = torch.FloatTensor(a_scaled).to(device)
    t_tensor = torch.FloatTensor(t_scaled).to(device)
    
    with torch.no_grad():
        preds = global_models['dl_model'](a_tensor, t_tensor)
        class_idx = preds.argmax(dim=1).item()
        
    predicted_label = global_models['le'].inverse_transform([class_idx])[0]
    print(f"  -> Predicted Emotion Tag: {predicted_label}")
    
    # Map back to Edge-TTS friendly strings
    mapping = {
        "HAP": "happy", "SAD": "sad", "FEA": "fear", "ANG": "angry", 
        "NEU": "neutral", "DIS": "angry", "CAL": "neutral", "SUR": "happy"
    }
    return mapping.get(predicted_label, "neutral")

# ── LIVE RECORDING & APPROVAL ─────────────────────────────────
def record_audio(max_seconds: int = MAX_RECORD_SECS) -> str:
    os.makedirs(AUDIO_INPUT_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = os.path.join(AUDIO_INPUT_DIR, f"input_{timestamp}.wav")

    print(f"\n[MIC] Recording ... speak now. Press ENTER to stop.")
    frames, recording = [], True

    def _record():
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
            while recording:
                data, _ = stream.read(SAMPLE_RATE // 10)
                frames.append(data.copy())

    t = threading.Thread(target=_record, daemon=True)
    t.start()
    input()
    recording = False
    t.join(timeout=1)

    if not frames:
        return ""

    audio = np.concatenate(frames, axis=0).flatten()[:SAMPLE_RATE * max_seconds]
    sf.write(filename, audio, SAMPLE_RATE)
    return filename

def approval_gate(transcript: str) -> tuple[bool, str]:
    print("\n" + "─" * 58)
    print("  ASR TRANSCRIPT (what Whisper heard):")
    print(f"\n    \"{transcript}\"\n")
    print("  Is this correct?")
    print("  [y] Yes, use this         (proceed to generation)")
    print("  [e] Edit it manually      (correct misheard words)")
    print("  [n] No, discard and retry (re-record)")
    print("─" * 58)
    while True:
        choice = input("  Your choice: ").strip().lower()
        if choice == "y": return True, transcript
        elif choice == "e":
            edited = input("  Enter corrected transcript: ").strip()
            if edited: return True, edited
            return True, transcript
        elif choice == "n": return False, ""
        else: print("  [!] Please enter y, e, or n.")

# ── TTS OUTPUT ────────────────────────────────────────────────

# ── EXPANDED PROSODY MAP ──────────────────────────────────────────────────────
# Covers all 8 TORGO/SSI emotion labels and the internal 'tired' alias.
# Prosody values tested against edge-tts en-US-AnaNeural.
EMOTION_PROSODY = {
    # Core child-facing emotions
    "happy":    {"rate": "+15%", "pitch": "+20Hz", "volume": "+0%"},
    "sad":      {"rate": "-20%", "pitch": "-15Hz", "volume": "-10%"},
    "angry":    {"rate": "+10%", "pitch": "-10Hz", "volume": "+25%"},
    "neutral":  {"rate": "+0%",  "pitch": "+0Hz",  "volume": "+0%"},
    # Aliases used by Mistral/clinical output
    "tired":    {"rate": "-30%", "pitch": "-25Hz", "volume": "-20%"},
    # Additional TORGO / SSI dataset labels (now routed via SVM)
    "ang":      {"rate": "+10%", "pitch": "-10Hz", "volume": "+25%"},   # ANG
    "hap":      {"rate": "+15%", "pitch": "+20Hz", "volume": "+0%"},    # HAP
    "neu":      {"rate": "+0%",  "pitch": "+0Hz",  "volume": "+0%"},    # NEU
    "sad":      {"rate": "-20%", "pitch": "-15Hz", "volume": "-10%"},   # SAD
    "cal":      {"rate": "-10%", "pitch": "-5Hz",  "volume": "-5%"},    # CAL (calm)
    "dis":      {"rate": "-5%",  "pitch": "-5Hz",  "volume": "+5%"},    # DIS (disgust)
    "fea":      {"rate": "-10%", "pitch": "+15Hz", "volume": "+10%"},   # FEA (fear)
    "sur":      {"rate": "+20%", "pitch": "+25Hz", "volume": "+15%"},   # SUR (surprise)
    # Friendly aliases for the above
    "calm":     {"rate": "-10%", "pitch": "-5Hz",  "volume": "-5%"},
    "fear":     {"rate": "-10%", "pitch": "+15Hz", "volume": "+10%"},
    "disgust":  {"rate": "-5%",  "pitch": "-5Hz",  "volume": "+5%"},
    "surprise": {"rate": "+20%", "pitch": "+25Hz", "volume": "+15%"},
}

def clean_text_for_tts(text: str) -> str:
    text = re.sub(r"[\"\'*•\-]", "", text)
    text = re.sub(r"\[.*?\]", "", text).strip()
    return text + "." if text and not text.endswith(".") else text

async def _synthesize_async(text: str, voice: str, filename: str, prosody: dict):
    communicate = edge_tts.Communicate(
        text=text, voice=voice,
        rate=prosody.get("rate", "+0%"),
        pitch=prosody.get("pitch", "+0%"),
        volume=prosody.get("volume", "+0%")
    )
    await communicate.save(filename)

def synthesize_and_play(text: str, emotion: str = "neutral", label: str = "output") -> str:
    os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
    timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename   = os.path.join(AUDIO_OUTPUT_DIR, f"{label}_{timestamp}.mp3")

    cleaned = clean_text_for_tts(text)
    if not cleaned: return ""

    prosody = EMOTION_PROSODY.get(emotion.lower(), EMOTION_PROSODY["neutral"])
    print(f"\n[TTS] Emotion : {emotion.upper()} | Synthesizing: \"{cleaned}\"")

    asyncio.run(_synthesize_async(cleaned, TTS_VOICE, filename, prosody))
    
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    return filename

# ── CORE PIPELINE ─────────────────────────────────────────────

def run_aac_pipeline(
    whisper_model,
    kg_agent,
    guard_agent,
    context:    str,
    partner:    str,
    aac_agent   = None,
    audio_path: str  = "",
    svm_models: dict = None,
) -> tuple:
    """
    Full pipeline (v2.1):
      audio → ASR → approval → KG context
      → Context-Engineered SVM emotion (primary)
      → Mistral phrase generation (emotion tag used as fallback)
      → TTS with matched prosody
    Returns (aac_agent, success: bool).

    svm_models: dict with keys scaler, pca, clf, le — loaded at startup.
                If None, SVM emotion step is skipped (Mistral tag used instead).
    """

    # ── 1. Get audio ──────────────────────────────────────────
    if not audio_path:
        print("\n  [1] Record live | [2] Provide path")
        src = input("  Choice: ").strip()
        if src == "1": audio_path = record_audio()
        elif src == "2": audio_path = input("  Path: ").strip().strip('"')
        if not audio_path or not os.path.exists(audio_path): return aac_agent, False

    # ASR & Audio Vector Extraction
    transcript, w_vec, wh_vec, e_vec = transcribe_and_extract(audio_path)
    if not transcript: return aac_agent, False

    # Approval Gate
    approved, final_transcript = approval_gate(transcript)
    if not approved: return aac_agent, False

    # Deep Learning Emotion Inference
    final_emotion = predict_emotion(w_vec, wh_vec, e_vec, final_transcript)

    # KG Context
    kg_context_str = kg_agent.get_context(context, partner)
    linguistic_age = kg_agent.get_patient_linguistic_age()
    
    # Mistral AAC Generation (Strict JSON)
    if aac_agent is None: aac_agent = OllamaMistralAgent(kg_context_str, linguistic_age)
    else: aac_agent.update_context(kg_context_str, linguistic_age)

    print("\n[OllamaMistralAgent] Generating deterministic JSON phrase...")
    prompt = (
        f"Aarav said (fragmented): \"{final_transcript}\"\n"
        f"Context: {context}, talking to {partner}.\n"
        f"Detected Emotion: {final_emotion}\n"
    )
    result_json = aac_agent.step(prompt)
    clean_generated = result_json.get("phrase", "I am not sure.")

    generated = aac_agent.step(prompt)

    # Parse Mistral's inferred emotion tag (used as secondary / fallback)
    mistral_emotion = "neutral"
    match = re.search(r"\[(.*?)\]", generated)
    if match:
        mistral_emotion = match.group(1).lower()

    clean_generated = re.sub(r"\[.*?\]", "", generated).strip()

    # ── SVM Context-Engineered Emotion (primary) ──────────────────────────
    svm_emotion = None
    if svm_models:
        print("\n[ContextSVM] Predicting emotion with context-engineered SVM ...")
        try:
            import librosa, torch
            from transformers import WavLMModel, WhisperForConditionalGeneration, WhisperProcessor
            import opensmile

            # NOTE: For the live pipeline we reuse the already-transcribed text
            # from Whisper (final_transcript) and compute the LLM semantic vector
            # using the live KG context — the most accurate signal available.
            llm_vec = get_semantic_vector_with_context(
                transcription    = final_transcript,
                kg_context_str   = kg_context_str,
                scenario         = context,
                partner          = partner,
                temperature      = 0.3,
            )

            # Build a partial fusion vector using only the LLM component.
            # (WavLM + Whisper encoder are heavy to re-load live; the LLM vector
            #  alone already carries the context-engineered signal.)
            # For full accuracy, run rehabilitate_speech.py which loads all models.
            partial_vec   = llm_vec.reshape(1, -1)

            # Pad to match the trained scaler's expected feature dimension
            # by zero-filling the acoustic channels (wavlm + whisper + egemaps)
            scaler_n_feat = svm_models["scaler"].n_features_in_
            if partial_vec.shape[1] < scaler_n_feat:
                pad = np.zeros((1, scaler_n_feat - partial_vec.shape[1]), dtype=np.float32)
                # Place LLM vector in its correct position (after wavlm 768 + whisper 512)
                wavlm_zeros   = np.zeros((1, 768),  dtype=np.float32)
                whisper_zeros = np.zeros((1, 512),  dtype=np.float32)
                egemaps_zeros = np.zeros((1, 88),   dtype=np.float32)
                full_vec = np.concatenate(
                    [wavlm_zeros, whisper_zeros, llm_vec.reshape(1, -1), egemaps_zeros],
                    axis=1
                )
            else:
                full_vec = partial_vec

            scaled = svm_models["scaler"].transform(full_vec)
            pca_d  = svm_models["pca"].transform(scaled)
            pred   = svm_models["clf"].predict(pca_d)[0]
            svm_emotion = svm_models["le"].inverse_transform([pred])[0].lower()
            print(f"[ContextSVM] Predicted emotion: {svm_emotion.upper()}")
        except Exception as e:
            print(f"[ContextSVM] Error: {e} — falling back to Mistral emotion tag.")
            svm_emotion = None

    # Choose primary emotion: SVM (context-grounded) wins; Mistral is fallback
    emotion = svm_emotion if svm_emotion else mistral_emotion

    # ── 7. Clinical guard ─────────────────────────────────────
    print("\n[ClinicalGuardAgent] Validating ...")
    validation = guard_agent.validate(generated)

    print("\n── RESULT ────────────────────────────────────────────────")
    print(f"  Dysarthric input  : \"{final_transcript}\"")
    if svm_models and svm_emotion:
        print(f"  SVM Emotion       : {svm_emotion.upper()}  ← context-engineered (PRIMARY)")
        print(f"  Mistral Emotion   : {mistral_emotion.upper()}  (secondary / reference)")
    else:
        print(f"  Inferred Emotion  : {emotion.upper()}  (Mistral tag — SVM not loaded)")
    print(f"  Final Emotion     : {emotion.upper()}")
    print(f"  Structured output : \"{clean_generated}\"")
    print(f"  Guard             : {validation['status']}")
    if validation["violations"]:
        for v in validation["violations"]:
            print(f"    ✗  {v}")
    if validation["warnings"]:
        for w in validation["warnings"]:
            print(f"    ⚠  {w}")

    # ── 8. TTS synthesis + playback ───────────────────────────
    if validation["status"] == "FAIL":
        print("\n  [!] Guard FAILED — synthesizing for demo.")

    audio_out = synthesize_and_play(clean_generated, emotion=emotion, label=f"{context}_{partner}")

    if audio_out:
        print(f"\n── AUDIO OUTPUT ──────────────────────────────────────────")
        print(f"  Input audio  : {audio_path}")
        print(f"  Output audio : {audio_out}")
        print(f"  Folder       : {os.path.abspath(AUDIO_OUTPUT_DIR)}")
    print("──────────────────────────────────────────────────────────")

    # ── 9. Persist to KG ──────────────────────────────────────
    kg_agent.kg.add_episode(
        context    = context,
        partner    = partner,
        utterances = clean_generated,
        mcds       = 0.60,  # placeholder baseline
        success    = (validation["status"] == "PASS"),
        emotion    = emotion
    )

    # TTS Synthesis
    audio_out = synthesize_and_play(clean_generated, emotion=final_emotion, label=f"{context}")

    # Persist
    kg_agent.kg.add_episode(context=context, partner=partner, utterances=clean_generated, mcds=0.60, success=(validation["status"] == "PASS"), emotion=final_emotion)
    return aac_agent, True

# ── MAIN ──────────────────────────────────────────────────────

def load_svm_pipeline(model_save_dir: str = "saved_models") -> dict | None:
    """
    Load the pre-trained SVM pipeline from disk.
    Returns a dict with keys: scaler, pca, clf, le.
    Returns None if models are not found (SVM step will be skipped).
    """
    required = ["scaler.pkl", "pca.pkl", "svm_classifier.pkl", "label_encoder.pkl"]
    if not all(os.path.exists(os.path.join(model_save_dir, f)) for f in required):
        print(
            "[ContextSVM] Saved models not found in 'saved_models/'.\n"
            "             Run benchmark_multimodal.py then train_fusion.py to train them.\n"
            "             Emotion will fall back to Mistral's inferred tag."
        )
        return None

    return {
        "scaler": joblib.load(os.path.join(model_save_dir, "scaler.pkl")),
        "pca":    joblib.load(os.path.join(model_save_dir, "pca.pkl")),
        "clf":    joblib.load(os.path.join(model_save_dir, "svm_classifier.pkl")),
        "le":     joblib.load(os.path.join(model_save_dir, "label_encoder.pkl")),
    }


def main():
    print("=" * 60)
    print("  CP AAC Pipeline v2.1 — Context-Engineered Emotion")
    print(f"  ASR   : Whisper ({WHISPER_MODEL})")
    print(f"  LLM   : Mistral via Ollama")
    print(f"  Voice : {TTS_VOICE}")
    print(f"  Emotion: Context-Engineered SVM + Mistral fallback")
    print("=" * 60)
    
    try:
        requests.get(f"{OLLAMA_CHAT_URL.replace('/api/chat','')}/api/tags", timeout=2)
    except:
        print("[ERROR] Ollama not running.")
        return

    print("\n[OK] Ollama running.")
    print("[  ] Loading Whisper ASR model ...")
    whisper_model = load_whisper(WHISPER_MODEL)

    print("[  ] Loading SVM emotion pipeline ...")
    svm_models = load_svm_pipeline()
    if svm_models:
        print("[OK] SVM pipeline loaded (context-engineered emotion active).")

    print("[  ] Connecting to Neo4j ...")
    kg_agent    = KGContextAgent()
    guard_agent = ClinicalGuardAgent(kg_agent)
    aac_agent   = None

    print("\n[OK] All systems ready.")
    print(f"[  ] Input audio  saved to : {os.path.abspath(AUDIO_INPUT_DIR)}")
    print(f"[  ] Output audio saved to : {os.path.abspath(AUDIO_OUTPUT_DIR)}\n")

    while True:
        print("\n── MAIN MENU ─────────────────────────────────────────────")
        for k, v in SCENARIOS.items(): print(f"  [{k}] {v['context']:10s} | Partner: {v['partner']}")
        choice = input("\nYour choice (q to quit): ").strip().lower()

        if choice == "q":
            kg_agent.close()
            pygame.mixer.quit()
            break

        elif choice == "g":
            show_kg_summary(kg_agent)

        elif choice == "r":
            if aac_agent:
                aac_agent.reset_history()
                print("\n[OK] Conversation history cleared.")
            else:
                print("\n[OK] No history yet.")

        elif choice == "c":
            context = input("  Context (morning/school/therapy/physio/evening): ").strip()
            partner = input("  Partner (Priya/Vijay/Rohan/Dr. Meera/Dr. Sharma): ").strip()
            aac_agent, _ = run_aac_pipeline(
                whisper_model, kg_agent, guard_agent,
                context, partner, aac_agent,
                svm_models=svm_models,
            )

        elif choice in SCENARIOS:
            s = SCENARIOS[choice]
            aac_agent, _ = run_aac_pipeline(
                whisper_model, kg_agent, guard_agent,
                s["context"], s["partner"], aac_agent,
                svm_models=svm_models,
            )

        else:
            print("[!] Invalid choice.")


if __name__ == "__main__":
    main()
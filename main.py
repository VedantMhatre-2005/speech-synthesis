"""
CP Speech Synthesis — Full AAC Pipeline (Deep Learning Edition)
Dysarthric audio -> Whisper ASR -> User approval
-> Multimodal DL Emotion Inference -> Edge-TTS Audio Output

Prerequisites:
    1. Neo4j Desktop running
    2. ollama serve 
    3. python train_fusion.py (to generate models in saved_models/)
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
from transformers import WavLMModel, WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
from agents import KGContextAgent, OllamaMistralAgent, ClinicalGuardAgent

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
EMOTION_PROSODY = {
    "happy":   {"rate": "+15%", "pitch": "+20Hz", "volume": "+0%"},
    "sad":     {"rate": "-20%", "pitch": "-15Hz", "volume": "-10%"},
    "angry":   {"rate": "+10%", "pitch": "-10Hz", "volume": "+25%"},
    "neutral": {"rate": "+0%",  "pitch": "+0Hz",  "volume": "+0%"},
    "fear":    {"rate": "+25%", "pitch": "+30Hz", "volume": "-10%"},
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
def run_aac_pipeline(kg_agent, guard_agent, context: str, partner: str, aac_agent, audio_path: str = ""):
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

    # Clinical Guard Validation
    validation = guard_agent.validate(clean_generated)
    print("\n── RESULT ────────────────────────────────────────────────")
    print(f"  Input       : \"{final_transcript}\"")
    print(f"  DL Emotion  : {final_emotion.upper()}")
    print(f"  LLM Output  : \"{clean_generated}\"")
    print(f"  Guard       : {validation['status']}")

    # TTS Synthesis
    audio_out = synthesize_and_play(clean_generated, emotion=final_emotion, label=f"{context}")

    # Persist
    kg_agent.kg.add_episode(context=context, partner=partner, utterances=clean_generated, mcds=0.60, success=(validation["status"] == "PASS"), emotion=final_emotion)
    return aac_agent, True

# ── MAIN ──────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  CP AAC Pipeline — DEEP LEARNING EDITION")
    print("=" * 60)
    
    try:
        requests.get(f"{OLLAMA_CHAT_URL.replace('/api/chat','')}/api/tags", timeout=2)
    except:
        print("[ERROR] Ollama not running.")
        return

    load_all_models()
    kg_agent, guard_agent, aac_agent = KGContextAgent(), ClinicalGuardAgent(KGContextAgent()), None

    while True:
        print("\n── MAIN MENU ─────────────────────────────────────────────")
        for k, v in SCENARIOS.items(): print(f"  [{k}] {v['context']:10s} | Partner: {v['partner']}")
        choice = input("\nYour choice (q to quit): ").strip().lower()

        if choice == "q":
            kg_agent.close()
            pygame.mixer.quit()
            break
        elif choice in SCENARIOS:
            s = SCENARIOS[choice]
            aac_agent, _ = run_aac_pipeline(kg_agent, guard_agent, s["context"], s["partner"], aac_agent)

if __name__ == "__main__":
    main()
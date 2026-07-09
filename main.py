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
import asyncio
import requests
import datetime
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
import pygame
import whisper
import edge_tts
import joblib
from agents import KGContextAgent, OllamaMistralAgent, ClinicalGuardAgent
from context_engineering import get_semantic_vector_with_context

# ── CONFIG ────────────────────────────────────────────────────
AUDIO_OUTPUT_DIR  = "audio_outputs"
AUDIO_INPUT_DIR   = "audio_inputs"
TTS_VOICE         = "en-US-AnaNeural"   # child neural voice
WHISPER_MODEL     = "base"              # tiny / base / small / medium
                                        # base is best balance for dysarthric speech
SAMPLE_RATE       = 16000               # Whisper expects 16kHz
MAX_RECORD_SECS   = 15                  # safety ceiling for live recording

SCENARIOS = {
    "1": {"context": "morning",  "partner": "Priya"},
    "2": {"context": "therapy",  "partner": "Dr. Meera"},
    "3": {"context": "school",   "partner": "Rohan"},
    "4": {"context": "evening",  "partner": "Vijay"},
    "5": {"context": "physio",   "partner": "Dr. Sharma"},
}


# ── WHISPER ASR ───────────────────────────────────────────────

def load_whisper(model_name: str = WHISPER_MODEL):
    """
    Loads Whisper model onto GPU if available, else CPU.
    Downloads once (~150MB for base), cached after.
    """
    print(f"\n[Whisper] Loading model: '{model_name}' ...")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = whisper.load_model(model_name, device=device)
    print(f"[Whisper] Ready on {device.upper()}.")
    return model


def transcribe_audio(whisper_model, audio_path: str) -> str:
    """
    Transcribes a WAV/MP3 file using Whisper.
    Uses a prompt that biases Whisper toward dysarthric child speech patterns —
    short fragments, missing function words, simplified grammar.
    """
    print(f"\n[Whisper] Transcribing: {audio_path}")

    result = whisper_model.transcribe(
        audio_path,
        language="en",
        # Initial prompt biases Whisper toward fragmented child speech
        initial_prompt=(
            "Child speech with cerebral palsy. "
            "Short phrases. Incomplete sentences. "
            "Words may be unclear or fragmented."
        ),
        temperature=0.0,        # greedy decoding — more deterministic
        best_of=1,
        fp16=False,             # safer across hardware
    )

    transcript = result["text"].strip()
    print(f"[Whisper] Raw transcript: \"{transcript}\"")
    return transcript


# ── LIVE RECORDING ────────────────────────────────────────────

def record_audio(max_seconds: int = MAX_RECORD_SECS) -> str:
    """
    Records from the default microphone until the user presses Enter.
    Saves to audio_inputs/ and returns the file path.
    """
    os.makedirs(AUDIO_INPUT_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = os.path.join(AUDIO_INPUT_DIR, f"input_{timestamp}.wav")

    print(f"\n[MIC] Recording ... speak now.")
    print(f"      Press ENTER to stop (max {max_seconds}s).")

    frames     = []
    recording  = True

    def _record():
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                             dtype="float32") as stream:
            while recording:
                data, _ = stream.read(SAMPLE_RATE // 10)  # 100ms chunks
                frames.append(data.copy())

    # Record in background thread so Enter press can stop it
    t = threading.Thread(target=_record, daemon=True)
    t.start()
    input()                # blocks until Enter
    recording = False
    t.join(timeout=1)

    if not frames:
        print("[MIC] No audio captured.")
        return ""

    audio = np.concatenate(frames, axis=0).flatten()

    # Enforce max duration
    max_samples = SAMPLE_RATE * max_seconds
    audio = audio[:max_samples]

    sf.write(filename, audio, SAMPLE_RATE)
    duration = len(audio) / SAMPLE_RATE
    print(f"[MIC] Recorded {duration:.1f}s → {filename}")
    return filename


# ── APPROVAL GATE ─────────────────────────────────────────────

def approval_gate(transcript: str) -> tuple[bool, str]:
    """
    Shows the ASR transcript to the user and asks for approval.
    The user can:
      [y]        approve as-is
      [e]        edit the transcript manually
      [n]        discard and start over
    Returns (approved: bool, final_transcript: str).
    """
    print("\n" + "─" * 58)
    print("  ASR TRANSCRIPT (what Whisper heard):")
    print(f"\n    \"{transcript}\"\n")
    print("  Is this correct?")
    print("  [y] Yes, use this         (proceed to generation)")
    print("  [e] Edit it manually      (correct misheard words)")
    print("  [n] No, discard and retry (re-record or re-enter)")
    print("─" * 58)

    while True:
        choice = input("  Your choice: ").strip().lower()

        if choice == "y":
            return True, transcript

        elif choice == "e":
            print(f"\n  Current: \"{transcript}\"")
            edited = input("  Enter corrected transcript: ").strip()
            if edited:
                print(f"\n  Updated transcript: \"{edited}\"")
                confirm = input("  Confirm? [y/n]: ").strip().lower()
                if confirm == "y":
                    return True, edited
            else:
                print("  [!] Empty input. Keeping original.")
                return True, transcript

        elif choice == "n":
            return False, ""

        else:
            print("  [!] Please enter y, e, or n.")


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
    text    = re.sub(r"[\"\'*•\-]", "", text)
    text    = re.sub(r"\[.*?\]",    "", text)
    text    = re.sub(r"\(.*?\)",    "", text)
    phrases = re.split(r"[\n,]+", text)
    phrases = [p.strip() for p in phrases if p.strip()]
    cleaned = ". ".join(phrases)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned and not cleaned.endswith("."):
        cleaned += "."
    return cleaned


async def _synthesize_async(text: str, voice: str, filename: str, prosody: dict):
    communicate = edge_tts.Communicate(
        text=text, 
        voice=voice,
        rate=prosody.get("rate", "+0%"),
        pitch=prosody.get("pitch", "+0%"),
        volume=prosody.get("volume", "+0%")
    )
    await communicate.save(filename)


def synthesize_and_play(text: str, emotion: str = "neutral", label: str = "output") -> str:
    os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
    timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = re.sub(r"\W+", "_", label)[:30]
    filename   = os.path.join(AUDIO_OUTPUT_DIR, f"{safe_label}_{timestamp}.mp3")

    cleaned = clean_text_for_tts(text)
    if not cleaned:
        print("[TTS] Nothing to synthesize.")
        return ""

    prosody = EMOTION_PROSODY.get(emotion.lower(), EMOTION_PROSODY["neutral"])

    print(f"\n[TTS] Voice      : {TTS_VOICE}")
    print(f"[TTS] Emotion    : {emotion.upper()}")
    print(f"[TTS] Prosody    : Rate={prosody['rate']}, Pitch={prosody['pitch']}, Vol={prosody['volume']}")
    print(f"[TTS] Synthesizing: \"{cleaned}\"")

    asyncio.run(_synthesize_async(cleaned, TTS_VOICE, filename, prosody))
    print(f"[TTS] Saved → {filename}")

    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    print("[TTS] Playing ...")
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
        print("\n  How would you like to provide your speech?")
        print("  [1] Record live from microphone")
        print("  [2] Provide path to an existing audio file")
        src = input("  Choice: ").strip()

        if src == "1":
            audio_path = record_audio()
            if not audio_path:
                print("[!] Recording failed. Aborting.")
                return aac_agent, False

        elif src == "2":
            audio_path = input("  Full path to audio file (.wav / .mp3): ").strip().strip('"')
            if not os.path.exists(audio_path):
                print(f"[!] File not found: {audio_path}")
                return aac_agent, False

        else:
            print("[!] Invalid choice.")
            return aac_agent, False

    # ── 2. ASR — Whisper transcription ────────────────────────
    transcript = transcribe_audio(whisper_model, audio_path)

    if not transcript:
        print("[!] Whisper returned empty transcript. Try again.")
        return aac_agent, False

    # ── 3. Approval gate ──────────────────────────────────────
    approved, final_transcript = approval_gate(transcript)

    if not approved:
        print("\n[Pipeline] Transcript rejected. Returning to menu.")
        return aac_agent, False

    print(f"\n[Pipeline] Approved transcript: \"{final_transcript}\"")

    # ── 4. KG context query ───────────────────────────────────
    print(f"\n[KGContextAgent] Querying Neo4j — context='{context}', partner='{partner}'")
    kg_context_str = kg_agent.get_context(context, partner)
    linguistic_age = kg_agent.get_patient_linguistic_age()

    print("\n── KG CONTEXT ────────────────────────────────────────────")
    print(kg_context_str)

    # ── 5. Build or update AAC agent ──────────────────────────
    if aac_agent is None:
        aac_agent = OllamaMistralAgent(kg_context_str, linguistic_age)
        print("\n[OllamaMistralAgent] Agent initialised.")
    else:
        aac_agent.update_context(kg_context_str, linguistic_age)
        print("\n[OllamaMistralAgent] Context updated.")

    # ── 6. Mistral generation ─────────────────────────────────
    print("[OllamaMistralAgent] Generating structured phrase ...")

    prompt = (
        f"Aarav said (fragmented dysarthric speech): \"{final_transcript}\"\n\n"
        f"Context: {context}, talking to {partner}.\n\n"
        "Reconstruct what Aarav meant as 1 to 3 short, clear, natural phrases "
        "a child would say. Use his preferred vocabulary from the context above. \n"
        "MANDATORY: Start with an emotion tag in brackets, e.g., [Happy]. \n"
        "Keep each phrase under 6 words. First person only. No explanation."
    )

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

    return aac_agent, True


# ── KG SUMMARY ────────────────────────────────────────────────

def show_kg_summary(kg_agent):
    summary = kg_agent.get_summary()
    print("\n── NEO4J KNOWLEDGE GRAPH SUMMARY ─────────────────────────")
    print("  Node counts:")
    for row in summary["nodes"]:
        print(f"    {row['label']:15s}: {row['count']}")
    print("\n  Relationship counts:")
    for row in summary["relationships"]:
        print(f"    {row['type']:25s}: {row['count']}")
    print("\n  Visualise at : http://localhost:7474")
    print("  Cypher       : MATCH (n) RETURN n")
    print("──────────────────────────────────────────────────────────")


# ── OLLAMA CHECK ──────────────────────────────────────────────

def check_ollama(host: str = "http://localhost:11434") -> bool:
    """
    Check if Ollama is running and accessible.
    Returns True if responding, False otherwise.
    """
    try:
        response = requests.get(f"{host}/api/tags", timeout=2)
        return response.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


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

    if not check_ollama():
        print("\n[ERROR] Ollama not running. Start with: ollama serve")
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
        print("  SPEAK AS AARAV — select context first, then provide audio\n")
        for k, v in SCENARIOS.items():
            print(f"  [{k}] {v['context']:10s} | Partner: {v['partner']}")
        print("  [c]   Custom context + partner")
        print("  [r]   Reset conversation history")
        print("  [g]   Show KG summary")
        print("  [q]   Quit")

        choice = input("\nYour choice: ").strip().lower()

        if choice == "q":
            kg_agent.close()
            pygame.mixer.quit()
            print("\n[EXIT] Goodbye.")
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
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
warnings.filterwarnings('ignore')

from transformers import WavLMModel, WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

# Add temp_openvoice to path for voice cloning
sys.path.append(os.path.abspath("../speech_synthesis_dataset/temp_openvoice"))
try:
    from openvoice.api import ToneColorConverter
except ImportError:
    print("[!] OpenVoice not found. Voice cloning step will be skipped.")

# ── CONFIGURATION ─────────────────────────────────────────────
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.1:8b"

LORA_PATH = "../whisper-dysarthric-lora/checkpoint-500"
MODELS_DIR = "saved_models"
REFERENCE_DIR = "../speech_synthesis_dataset/torgo_references"

def load_ai_models():
    print(">>> Loading AI Models (This takes a moment)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Acoustic extractors
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(device)
    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals)
    
    # Fine-Tuned Whisper
    whisper_proc = WhisperProcessor.from_pretrained("openai/whisper-small")
    base_whisper = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)
    whisper_model = PeftModel.from_pretrained(base_whisper, LORA_PATH)
    
    # Pre-trained SVM Pipeline
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    pca = joblib.load(os.path.join(MODELS_DIR, 'pca.pkl'))
    svm = joblib.load(os.path.join(MODELS_DIR, 'svm_classifier.pkl'))
    le = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.pkl'))
    
    return device, wavlm, smile, whisper_proc, whisper_model, scaler, pca, svm, le

def process_dysarthric_speech(audio_path, models):
    device, wavlm, smile, whisper_proc, whisper_model, scaler, pca, svm, le = models
    print(f"\n[1] Processing Patient Audio: {audio_path}")
    
    arr, sr = librosa.load(audio_path, sr=16000)
    
    # Whisper Transcription
    f_in = whisper_proc(arr, sampling_rate=16000, return_tensors="pt").input_features.to(device)
    predicted_ids = whisper_model.generate(f_in, language="english", task="transcribe")
    transcription = whisper_proc.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    print(f"    [+] Decoded Transcript: '{transcription}'")
    
    # Whisper Acoustic
    encoder_outputs = whisper_model.get_encoder()(f_in)
    wh_out = encoder_outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    
    # WavLM
    ten = torch.tensor(arr).unsqueeze(0).to(device)
    w_out = wavlm(ten).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    
    # eGeMAPS
    egemaps_vec = smile.process_signal(arr, sr).values.flatten().astype(np.float32)
    
    # LLM Reasoning
    prompt = f'Text: "{transcription}"\nDescribe the likely emotional state of the speaker in 3-5 keywords. Output ONLY the keywords separated by commas.'
    res = requests.post(OLLAMA_CHAT_URL, json={"model": LLM_MODEL, "messages": [{"role": "user", "content": prompt}], "stream": False, "options": {"temperature": 0.3}})
    reasoning = res.json()["message"]["content"].strip()
    
    embed_res = requests.post(OLLAMA_EMBED_URL, json={"model": EMBED_MODEL, "input": reasoning})
    llm_vec = np.array(embed_res.json()["embeddings"][0], dtype=np.float32)
    
    # Classify Emotion
    print("[2] Predicting Emotion...")
    vec = np.concatenate([w_out, wh_out, llm_vec, egemaps_vec]).reshape(1, -1)
    vec_scaled = scaler.transform(vec)
    vec_pca = pca.transform(vec_scaled)
    pred_idx = svm.predict(vec_pca)[0]
    predicted_emotion = le.inverse_transform([pred_idx])[0]
    print(f"    [+] Detected Emotion: {predicted_emotion.upper()}")
    
    return transcription, predicted_emotion

def generate_rehabilitated_speech(transcription, emotion, output_path):
    print(f"\n[3] Generating Clear Speech (TTS)...")
    
    # Step 1: Base TTS (Generating clear English speech)
    # Note: In a true production app, you would feed the 'emotion' label into a controllable 
    # emotional TTS engine (like F5-TTS or Bark) here. For this demo, we generate clear base speech.
    temp_tts = "temp_clear_speech.wav"
    tts = gTTS(text=transcription, lang='en', slow=False)
    tts.save(temp_tts)
    print("    [+] Base clear speech generated.")
    
    # Step 2: Voice Cloning (OpenVoice)
    print("[4] Cloning Patient's Voice Profile...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_converter = '../speech_synthesis_dataset/checkpoints/converter/checkpoint.pth'
    config_converter = '../speech_synthesis_dataset/checkpoints/converter/config.json'
    
    try:
        tone_color_converter = ToneColorConverter(config_converter, device=device, enable_watermark=False)
        tone_color_converter.load_ckpt(ckpt_converter)
        
        # Load Patient Reference (Using F01 for this demo)
        patient_ref = os.path.join(REFERENCE_DIR, "F01_ref.wav")
        tgt_se = tone_color_converter.extract_se(patient_ref)
        src_se = tone_color_converter.extract_se(temp_tts)
        
        tone_color_converter.convert(
            audio_src_path=temp_tts, 
            src_se=src_se, 
            tgt_se=tgt_se, 
            output_path=output_path,
            message="@MyShell"
        )
        print(f"\n🎉 SUCCESS! Fully rehabilitated audio saved to: {output_path}")
        
    except Exception as e:
        print(f"    [!] Error during voice cloning: {e}")
        
    finally:
        if os.path.exists(temp_tts):
            os.remove(temp_tts)

if __name__ == "__main__":
    print("="*70)
    print(" END-TO-END DYSARTHRIC SPEECH REHABILITATION ")
    print("="*70)
    
    # Test on one of the dysarthric clips
    TEST_AUDIO = "../speech_synthesis_dataset/openvoice_output/F01/F01_clip_0_ANG.wav"
    OUTPUT_AUDIO = "rehabilitated_output.wav"
    
    models = load_ai_models()
    transcript, emotion = process_dysarthric_speech(TEST_AUDIO, models)
    generate_rehabilitated_speech(transcript, emotion, OUTPUT_AUDIO)

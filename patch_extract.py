import re

with open("run_ablation_datasets.py", "r") as f:
    content = f.read()

new_extract = """def extract_features(metadata_path, output_pt_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    unload_ollama_model()
    clear_gpu()
    
    chk_path = output_pt_path.replace(".pt", "_checkpoint.pt")
    base_data = []
    if os.path.exists(chk_path):
        try:
            base_data = torch.load(chk_path, weights_only=False)
            print(f"[!] Resuming acoustic extraction from checkpoint: {len(base_data)} samples already processed.")
        except:
            print("[!] Failed to load checkpoint. Starting from scratch.")
            
    df = pd.read_csv(metadata_path)
    
    if len(base_data) < len(df):
        print(f"\\n[+] Loading Models on {device}...")
        wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(device)
        whisper_proc = WhisperProcessor.from_pretrained("openai/whisper-small")
        base_whisper = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)
        try:
            whisper_model = PeftModel.from_pretrained(base_whisper, LORA_PATH)
        except:
            print("[!] Warning: LoRA not found, using base whisper.")
            whisper_model = base_whisper
            
        smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals)
        
        print(f"\\n[+] Processing {len(df)} Audio files from {metadata_path}...")
        with torch.no_grad():
            for i in tqdm(range(len(base_data), len(df)), initial=len(base_data), total=len(df)):
                row = df.iloc[i]
                audio_path = row['filepath']
                if not os.path.isabs(audio_path):
                    audio_path = os.path.join("../Dataset_Degradation", audio_path.replace("\\\\", "/"))
                
                if not os.path.exists(audio_path):
                    continue
                    
                try:
                    arr, sr = librosa.load(audio_path, sr=16000)
                except:
                    continue
                    
                ten = torch.tensor(arr).unsqueeze(0).to(device)
                w_out = wavlm(ten).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                
                f_in = whisper_proc(arr, sampling_rate=16000, return_tensors="pt").input_features.to(device)
                predicted_ids = whisper_model.generate(f_in, language="english", task="transcribe")
                transcription = whisper_proc.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                
                encoder_outputs = whisper_model.get_encoder()(f_in)
                wh_out = encoder_outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                
                egemaps_df = smile.process_signal(arr, sr)
                egemaps_vec = egemaps_df.values.flatten().astype(np.float32)
                
                base_data.append({
                    "label": row["emotion"],
                    "speaker": str(row.get("speaker_id", "UNK")),
                    "transcription": transcription,
                    "wavlm_vector": w_out,
                    "whisper_vector": wh_out,
                    "egemaps_vector": egemaps_vec
                })
                
                if (i + 1) % 250 == 0:
                    torch.save(base_data, chk_path)
            
            torch.save(base_data, chk_path)
            
        print("\\n[+] Unloading acoustic models...")
        del wavlm, whisper_model, base_whisper
        clear_gpu()
    else:
        print(f"[+] Acoustic features for all {len(df)} files already extracted.")
        
    print("\\n[+] LLM Emotional Extraction from Transcriptions...")
    llm_chk_path = output_pt_path.replace(".pt", "_llm_checkpoint.pt")
    final_recs = []
    if os.path.exists(llm_chk_path):
        try:
            final_recs = torch.load(llm_chk_path, weights_only=False)
            print(f"[!] Resuming LLM extraction from checkpoint: {len(final_recs)} samples processed.")
        except:
            pass

    if len(final_recs) < len(base_data):
        model_name = MODELS[0]
        for i in tqdm(range(len(final_recs), len(base_data)), initial=len(final_recs), total=len(base_data)):
            entry = base_data[i]
            reasoning = get_llm_reasoning(entry["transcription"], model_name)
            llm_vec = get_semantic_vector(reasoning)
            rec = entry.copy()
            rec["llm_vector"] = llm_vec
            final_recs.append(rec)
            
            if (i + 1) % 250 == 0:
                torch.save(final_recs, llm_chk_path)
                
        torch.save(final_recs, llm_chk_path)
        
    torch.save(final_recs, output_pt_path)
    print(f"  [OK] Saved final dataset -> {output_pt_path}")
    return output_pt_path
"""

# Extract everything before and after the extract_features function
pattern = re.compile(r'(def extract_features\(metadata_path, output_pt_path\):.*?)(?=def train_and_eval)', re.DOTALL)
new_content = pattern.sub(new_extract + "\n", content)

with open("run_ablation_datasets.py", "w") as f:
    f.write(new_content)
print("Patched successfully")

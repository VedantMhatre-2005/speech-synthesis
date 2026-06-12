import requests
import json

OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
MODELS = ["qwen2.5:7b", "llama3.1:8b", "mistral:7b", "qwen3:14B"]

def test_embeddings():
    print("="*60)
    print(" OLLAMA EMBEDDING DIAGNOSTIC")
    print("="*60)
    
    for model in MODELS:
        print(f"\nTesting Model: {model}")
        payload = {
            "model": model,
            "input": "This is a test transcript for emotion detection."
        }
        try:
            res = requests.post(OLLAMA_EMBED_URL, json=payload, timeout=30)
            if res.status_code == 200:
                data = res.json()
                if "embeddings" in data:
                    emb_len = len(data["embeddings"][0])
                    print(f"  [SUCCESS] Received vector of length: {emb_len}")
                else:
                    print(f"  [ERROR] Key 'embeddings' not found in response: {data}")
            else:
                print(f"  [FAILED] Status {res.status_code}: {res.text}")
        except Exception as e:
            print(f"  [EXCEPTION] {e}")

    print("\n" + "="*60)

if __name__ == "__main__":
    test_embeddings()

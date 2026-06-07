
import sys
import os
import re

# Add current dir to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents import OllamaMistralAgent, KGContextAgent, ClinicalGuardAgent
from main import synthesize_and_play, EMOTION_PROSODY

def test_emotion_inference():
    print("\n--- Phase 1: Testing LLM Emotion Inference ---")
    
    # Mock context
    mock_kg_context = """
    Patient: Aarav (6.0 years old)
    Routine: Therapy
    Partner: Dr. Meera
    Vocabulary: ['ouch', 'help', 'leg', 'tired', 'again']
    """
    
    agent = OllamaMistralAgent(mock_kg_context, 6.0)
    
    test_cases = [
        {"input": "leg ouch again", "expected": "sad/angry/tired"},
        {"input": "play again meera", "expected": "happy/neutral"},
    ]
    
    for case in test_cases:
        print(f"\nTesting Input: \"{case['input']}\"")
        response = agent.step(case["input"])
        print(f"Response: {response}")
        
        match = re.search(r"\[(.*?)\]", response)
        if match:
            print(f"  ✓ Inferred Emotion: {match.group(1)}")
        else:
            print(f"  ✗ FAILED: No emotion tag found in response.")
            print(f"    Raw Output: {repr(response)}")

def test_tts_prosody():
    print("\n--- Phase 2: Testing TTS Prosody Mapping ---")
    
    test_emotions = ["happy", "sad", "angry", "tired"]
    text = "I want to play again."
    
    for emotion in test_emotions:
        print(f"\nTesting TTS for Emotion: {emotion.upper()}")
        prosody = EMOTION_PROSODY.get(emotion, EMOTION_PROSODY["neutral"])
        print(f"Expected Prosody: {prosody}")
        
        # We don't necessarily need to play it during automated tests, 
        # but we'll show the params that WOULD be used.
        filename = f"test_output_{emotion}.mp3"
        print(f"Synthesis call: synthesize_and_play(\"{text}\", emotion=\"{emotion}\")")

def main():
    print("="*50)
    print(" AAC Pipeline Verification Tool")
    print("="*50)
    
    try:
        test_emotion_inference()
        test_tts_prosody()
        print("\n" + "="*50)
        print("VERIFICATION COMPLETE")
        print("1. Check if LLM inferred appropriate emotions.")
        print("2. Listen to generated audio in audio_outputs/ (if running full main).")
        print("="*50)
    except Exception as e:
        print(f"\n[ERROR] Verification failed: {str(e)}")

if __name__ == "__main__":
    main()

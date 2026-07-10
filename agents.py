"""
Camel AI agent definitions — Ollama backend + Neo4j KG.
"""

import requests
import re
from knowledge_graph import Neo4jKG, format_context_for_prompt

OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.1:8b"


# ── 1. KG CONTEXT AGENT ───────────────────────────────────────

class KGContextAgent:
    """
    Owns all Neo4j Knowledge Graph operations.
    """

    def __init__(self):
        self.kg = Neo4jKG()
        # Build the graph on first run; comment this out on subsequent
        # runs if you want to preserve existing Neo4j data.
        self.kg.build()

    def get_context(self, current_context: str, partner: str) -> str:
        context_dict = self.kg.query_context(current_context, partner)
        return format_context_for_prompt(context_dict)

    def get_patient_linguistic_age(self) -> float:
        rows = self.kg.run("""
            MATCH (p:Patient {name: 'Aarav'})
            RETURN p.linguistic_age AS la
        """)
        return rows[0]["la"] if rows else 6.0

    def get_preferred_vocab(self) -> list[str]:
        rows = self.kg.run("""
            MATCH (:Patient {name: 'Aarav'})-[:PREFERS_VOCAB]->(v:Vocabulary)
            RETURN v.preferred_form AS pf
        """)
        return [r["pf"] for r in rows]

    def get_summary(self) -> dict:
        return self.kg.get_summary()

    def close(self):
        self.kg.close()


# ── 2. OLLAMA MISTRAL AGENT (JSON & FEW-SHOT) ─────────────────

class OllamaMistralAgent:
    """
    Sends structured chat messages to Llama3.1 running inside Ollama.
    Maintains conversation history and forces strict JSON output.
    """

    def __init__(self, kg_context_str: str, linguistic_age: float):
        self.history       = []
        self.update_context(kg_context_str, linguistic_age)

    def step(self, user_message: str) -> dict:
        self.history.append({"role": "user", "content": user_message})
        trimmed = self.history[-10:]

        payload = {
            "model":    OLLAMA_MODEL,
            "messages": [{"role": "system", "content": self.system_prompt}]
                        + trimmed,
            "stream":   False,
            "format":   "json",
            "options":  {"temperature": 0.2, "num_predict": 150},
        }

        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=120)
            response.raise_for_status()
            res_json = response.json()
            if "message" in res_json and "content" in res_json["message"]:
                reply = res_json["message"]["content"].strip()
            else:
                reply = '{"phrase": "[ERROR] Unexpected Ollama response"}'
        except Exception as e:
            reply = f'{{"phrase": "[ERROR] {str(e)}"}}'

        self.history.append({"role": "assistant", "content": reply})
        import json
        try:
            return json.loads(reply)
        except:
            return {"phrase": reply}

    def update_context(self, kg_context_str: str, linguistic_age: float):
        """Call this when context changes (new scenario) but history should persist."""
        self.system_prompt = (
            "You are an AAC assistant for Aarav, a child with cerebral palsy. "
            f"Aarav's linguistic age is {linguistic_age:.1f} years. \n"
            "MANDATORY: You MUST respond ONLY with a valid JSON object. \n"
            "Format: {\"phrase\": \"<reconstructed text>\"} \n"
            "Max 5 words per phrase. No commentary.\n\n"
            "FEW-SHOT EXAMPLES:\n"
            "User: Context: morning. Speech: \"n-nn-eed m-i-lk\"\n"
            "Assistant: {\"phrase\": \"I need milk please.\"}\n"
            "User: Context: therapy. Speech: \"h-h-urts l-e-g\"\n"
            "Assistant: {\"phrase\": \"My leg hurts a lot.\"}\n\n"
            + kg_context_str
        )

    def reset_history(self):
        self.history = []


# ── 3. CLINICAL GUARD AGENT ───────────────────────────────────

class ClinicalGuardAgent:
    """
    Validates generated output against KG-derived constraints.
    """

    def __init__(self, kg_agent: KGContextAgent):
        self.preferred_vocab = kg_agent.get_preferred_vocab()
        self.linguistic_age  = kg_agent.get_patient_linguistic_age()
        self.max_syllables   = 3

    def _count_syllables(self, word: str) -> int:
        word       = word.lower().strip(".,!?\"'")
        count      = 0
        vowels     = "aeiouy"
        prev_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        return max(1, count)

    def validate(self, generated_text: str) -> dict:
        violations, warnings_ = [], []
        
        # Strip [Emotion] tag for validation
        clean_text = re.sub(r"\[.*?\]", "", generated_text).strip()
        words = clean_text.split()

        if not words:
            violations.append("Output is empty after removing emotion tags.")

        if len(words) > 25:
            violations.append(f"Output too long ({len(words)} words).")

        for word in words:
            if self._count_syllables(word) > self.max_syllables:
                warnings_.append(
                    f"'{word}' may exceed linguistic age {self.linguistic_age:.1f}."
                )

        lower = clean_text.lower()
        for flag in ["aarav", "he wants", "he says", "the patient", "the child"]:
            if flag in lower:
                violations.append(f"Third-person narration detected: '{flag}'.")

        for flag in ["as an aac", "i recommend", "suggest", "the assistant"]:
            if flag in lower:
                violations.append(f"Clinical commentary detected: '{flag}'.")

        return {
            "status":     "PASS" if not violations else "FAIL",
            "violations": violations,
            "warnings":   warnings_,
        }

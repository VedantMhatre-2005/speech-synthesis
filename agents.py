"""
Camel AI agent definitions — Ollama/Mistral backend + Neo4j KG.
"""

import requests
from knowledge_graph import Neo4jKG, format_context_for_prompt

OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "mistral"


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


# ── 2. OLLAMA MISTRAL AGENT ───────────────────────────────────

class OllamaMistralAgent:
    """
    Sends structured chat messages to Mistral running inside Ollama.
    Maintains conversation history (last 10 turns).
    """

    def __init__(self, kg_context_str: str, linguistic_age: float):
        self.history       = []
        self.system_prompt = (
            "You are an Augmentative and Alternative Communication (AAC) assistant "
            "for a child named Aarav who has cerebral palsy and dysarthric speech. "
            f"Aarav's linguistic age is {linguistic_age:.1f} years. "
            "Generate ONLY short, simple phrases that Aarav would say. "
            "Keep each phrase to 5 words or fewer. "
            "Use only the vocabulary listed in the patient context below. "
            "Output 1 to 3 phrases only. "
            "Do NOT explain. Do NOT add commentary. "
            "Speak as Aarav in first person.\n\n"
            + kg_context_str
        )

    def step(self, user_message: str) -> str:
        self.history.append({"role": "user", "content": user_message})
        trimmed = self.history[-10:]

        payload = {
            "model":    OLLAMA_MODEL,
            "messages": [{"role": "system", "content": self.system_prompt}]
                        + trimmed,
            "stream":   False,
            "options":  {"temperature": 0.7, "num_predict": 150},
        }

        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=60)
            response.raise_for_status()
            reply = response.json()["message"]["content"].strip()
        except requests.exceptions.ConnectionError:
            reply = "[ERROR] Ollama not running. Run: ollama serve"
        except requests.exceptions.Timeout:
            reply = "[ERROR] Timeout. Run: ollama run mistral"
        except Exception as e:
            reply = f"[ERROR] {str(e)}"

        self.history.append({"role": "assistant", "content": reply})
        return reply

    def update_context(self, kg_context_str: str, linguistic_age: float):
        """Call this when context changes (new scenario) but history should persist."""
        self.system_prompt = (
            "You are an AAC assistant for Aarav, a child with cerebral palsy. "
            f"Aarav's linguistic age is {linguistic_age:.1f} years. "
            "Generate ONLY 1–3 short first-person phrases he would say. "
            "Max 5 words per phrase. No commentary.\n\n"
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
        words = generated_text.split()

        if len(words) > 25:
            violations.append(f"Output too long ({len(words)} words).")

        for word in words:
            if self._count_syllables(word) > self.max_syllables:
                warnings_.append(
                    f"'{word}' may exceed linguistic age {self.linguistic_age:.1f}."
                )

        lower = generated_text.lower()
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
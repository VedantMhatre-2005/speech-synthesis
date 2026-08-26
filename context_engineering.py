"""
context_engineering.py — Context-Engineered LLM Reasoning for Emotion Classification

Problem:
    The bare LLM prompt ("What are the emotional keywords in this text?") produces
    generic embeddings that poorly separate emotion classes for dysarthric speech.
    A child saying "hurt" could mean pain, tiredness, or anger depending on context.

Solution:
    Inject rich situational scaffolding — scenario, partner, routine, past episodes,
    and preferred vocabulary — sourced from the Neo4j Knowledge Graph — before asking
    the LLM for emotional keyword reasoning.

    This produces richer, more discriminative semantic vectors that significantly
    improve SVM classification accuracy.

Public API:
    build_context_prompt(transcription, kg_context_str, scenario, partner)
        → str: a grounded few-shot reasoning prompt

    get_llm_reasoning_with_context(transcription, kg_context_str, scenario,
                                   partner, model_name, ollama_url)
        → str: emotional keyword string (comma-separated)

    get_semantic_vector_with_context(...)
        → np.ndarray (768d): nomic-embed-text embedding of the keywords

    build_context_prompt_from_parts(transcription, patient_profile, routine_info,
                                    partner_info, vocab_list, past_episodes)
        → str: used when raw KG components are available individually
"""

import re
import requests
import numpy as np

# ── DEFAULTS ──────────────────────────────────────────────────────────────────

OLLAMA_CHAT_URL  = "http://localhost:11434/api/chat"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
EMBED_MODEL      = "nomic-embed-text"
LLM_MODEL        = "llama3.1:8b"

# Maps dataset-style emotion labels to plain English for prompt readability
EMOTION_LABEL_GLOSSARY = {
    "ANG": "angry / frustrated",
    "HAP": "happy / excited",
    "NEU": "neutral / calm",
    "SAD": "sad / upset",
    "CAL": "calm / relaxed",
    "DIS": "disgusted / repelled",
    "FEA": "fearful / anxious",
    "SUR": "surprised / startled",
}

# Few-shot examples grounding the LLM in dysarthric child speech patterns
FEW_SHOT_EXAMPLES = [
    {
        "scenario": "morning",
        "partner":  "mother",
        "transcript": "hurt leg bad",
        "keywords": "pain, distressed, uncomfortable, tired, helpless",
    },
    {
        "scenario": "school",
        "partner":  "classmate",
        "transcript": "play now want",
        "keywords": "eager, happy, excited, anticipatory, energetic",
    },
    {
        "scenario": "therapy",
        "partner":  "speech therapist",
        "transcript": "tired stop please",
        "keywords": "fatigued, frustrated, overwhelmed, exhausted, sad",
    },
    {
        "scenario": "evening",
        "partner":  "father",
        "transcript": "hungry want eat",
        "keywords": "hungry, needy, mild frustration, impatient, neutral",
    },
    {
        "scenario": "physio",
        "partner":  "physiotherapist",
        "transcript": "no more hurts",
        "keywords": "pain, fearful, resistant, distressed, unhappy",
    },
]


# ── PROMPT BUILDER ────────────────────────────────────────────────────────────

def build_context_prompt(
    transcription:  str,
    kg_context_str: str   = "",
    scenario:       str   = "",
    partner:        str   = "",
) -> str:
    """
    Construct a grounded LLM reasoning prompt that injects Knowledge Graph
    context around a dysarthric transcription.

    Parameters
    ----------
    transcription   : Raw ASR / Whisper output of the dysarthric utterance.
    kg_context_str  : The formatted KG context string from format_context_for_prompt().
    scenario        : One of morning/school/therapy/physio/evening.
    partner         : Name of the communication partner (e.g. "Priya", "Dr. Meera").

    Returns
    -------
    A complete prompt string ready to send to the LLM.
    """
    lines = []

    # ── System framing ────────────────────────────────────────────────────────
    lines.append("You are an expert child psychologist and AAC (Augmentative and")
    lines.append("Alternative Communication) specialist, trained in interpreting")
    lines.append("dysarthric speech from children with cerebral palsy.")
    lines.append("")

    # ── Patient profile (brief) ───────────────────────────────────────────────
    lines.append("PATIENT PROFILE:")
    lines.append("  Name           : Aarav")
    lines.append("  Condition      : Cerebral Palsy (spastic diplegia)")
    lines.append("  Chronological age : 8 years")
    lines.append("  Linguistic age    : 6.2 years (speaks in short fragments)")
    lines.append("  AAC device        : Tobii Dynavox I-13")
    lines.append("")

    # ── Situational context ───────────────────────────────────────────────────
    if scenario or partner:
        lines.append("CURRENT SITUATION:")
        if scenario:
            lines.append(f"  Time/routine   : {scenario}")
        if partner:
            lines.append(f"  Talking to     : {partner}")
        lines.append("")

    # ── KG context block (routine + vocab + past episodes) ───────────────────
    if kg_context_str and kg_context_str.strip():
        # Extract the most relevant parts to keep prompt size manageable
        kg_lines = [l for l in kg_context_str.splitlines() if l.strip()]
        # Include lines related to common phrases, preferred vocab, and past utterances
        relevant_keywords = [
            "common phrase", "preferred", "past successful", "utterance",
            "activities", "routine", "people present", "frequency",
        ]
        filtered_kg = [
            l for l in kg_lines
            if any(kw.lower() in l.lower() for kw in relevant_keywords)
        ]
        if filtered_kg:
            lines.append("KNOWLEDGE GRAPH CONTEXT (from patient's history):")
            for l in filtered_kg[:12]:   # cap at 12 lines to avoid token overflow
                lines.append(f"  {l.strip()}")
            lines.append("")

    # ── Few-shot examples ─────────────────────────────────────────────────────
    lines.append("REFERENCE EXAMPLES (dysarthric fragments → emotional keywords):")
    for ex in FEW_SHOT_EXAMPLES[:3]:   # use 3 examples to balance context vs. tokens
        lines.append(
            f'  Scenario "{ex["scenario"]}", partner "{ex["partner"]}":'
        )
        lines.append(f'    Fragment : "{ex["transcript"]}"')
        lines.append(f'    Keywords : {ex["keywords"]}')
    lines.append("")

    # ── Task ─────────────────────────────────────────────────────────────────
    lines.append("TASK:")
    lines.append(
        f'Given the context above, Aarav said (severely dysarthric, may be fragmented):'
    )
    lines.append(f'  "{transcription}"')
    lines.append("")
    lines.append(
        "Identify the most likely emotional state behind this utterance. "
        "Consider the situation, who he is talking to, and his communication history."
    )
    lines.append("")
    lines.append(
        "Output ONLY 3 to 5 emotional keywords, comma-separated. "
        "No explanation. No preamble. Just the keywords."
    )

    return "\n".join(lines)


def build_context_prompt_from_parts(
    transcription:   str,
    patient_profile: dict = None,
    routine_info:    dict = None,
    partner_info:    dict = None,
    vocab_list:      list = None,
    past_episodes:   list = None,
) -> str:
    """
    Alternative builder when you have the raw KG component dicts rather than
    the pre-formatted string. Useful for fine-grained control.
    """
    scenario = routine_info.get("context_key", "") if routine_info else ""
    partner  = partner_info.get("name", "")         if partner_info else ""

    # Build a lightweight KG context string from parts
    kg_lines = []

    if routine_info:
        kg_lines.append(
            f"Current routine: {routine_info.get('time', '')} "
            f"@ {routine_info.get('location', '')} — "
            f"Activities: {routine_info.get('activities', '')}"
        )
        kg_lines.append(
            f"Common phrases: {routine_info.get('common_utterances', '')}"
        )

    if vocab_list:
        kg_lines.append("Preferred vocabulary:")
        for v in (vocab_list or [])[:6]:
            kg_lines.append(
                f"  '{v.get('word', '')}' → \"{v.get('preferred_form', '')}\" "
                f"[{v.get('frequency', '')} frequency]"
            )

    if past_episodes:
        kg_lines.append("Past successful utterances in similar contexts:")
        for ep in (past_episodes or [])[:3]:
            kg_lines.append(f"  \"{ep.get('utterances', '')}\"")

    kg_context_str = "\n".join(kg_lines)

    return build_context_prompt(
        transcription=transcription,
        kg_context_str=kg_context_str,
        scenario=scenario,
        partner=partner,
    )


# ── LLM CALL ─────────────────────────────────────────────────────────────────

def get_llm_reasoning_with_context(
    transcription:  str,
    kg_context_str: str  = "",
    scenario:       str  = "",
    partner:        str  = "",
    model_name:     str  = LLM_MODEL,
    ollama_url:     str  = OLLAMA_CHAT_URL,
    temperature:    float = 0.3,
) -> str:
    """
    Call the Ollama LLM with a context-grounded prompt and return the
    emotional keyword string.

    Falls back to a bare-text prompt if the LLM call fails, so the pipeline
    always returns something usable.

    Returns
    -------
    str: comma-separated emotional keywords, e.g. "pain, distressed, tired"
    """
    prompt = build_context_prompt(
        transcription=transcription,
        kg_context_str=kg_context_str,
        scenario=scenario,
        partner=partner,
    )

    payload = {
        "model":   model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream":  False,
        "options": {"temperature": temperature, "num_predict": 80},
    }

    try:
        res = requests.post(ollama_url, json=payload, timeout=60)
        res.raise_for_status()
        content = res.json()["message"]["content"].strip()

        # Sanitize: keep only the keyword line (strip any stray explanation)
        content = _extract_keywords_only(content)
        return content

    except requests.exceptions.ConnectionError:
        print("[ContextEngineering] Ollama not running — falling back to bare prompt.")
        return _bare_fallback(transcription, model_name, ollama_url, temperature)

    except Exception as e:
        print(f"[ContextEngineering] LLM error: {e} — using fallback.")
        return "neutral, uncertain"


def _extract_keywords_only(text: str) -> str:
    """
    Given the LLM response, extract the first line that looks like a
    comma-separated keyword list and discard everything else.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in lines:
        # A keyword line has commas and no sentence-ending punctuation at the end
        if "," in line and not line.endswith((".", "?", "!")):
            # Strip any leading labels like "Keywords:" or "Output:"
            line = re.sub(r"^[A-Za-z\s]+:\s*", "", line)
            return line.strip()
    # Fallback: return the first non-empty line
    return lines[0] if lines else "neutral"


def _bare_fallback(
    transcription: str,
    model_name:    str,
    ollama_url:    str,
    temperature:   float,
) -> str:
    """Original bare-text prompt — used as a fallback if context call fails."""
    prompt = (
        f'Text: "{transcription}"\n'
        "Describe the likely emotional state of the speaker in 3-5 keywords. "
        "Output ONLY the keywords separated by commas."
    )
    payload = {
        "model":   model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream":  False,
        "options": {"temperature": temperature},
    }
    try:
        res = requests.post(ollama_url, json=payload, timeout=60)
        return res.json()["message"]["content"].strip()
    except Exception:
        return "neutral, calm"


# ── EMBEDDING ─────────────────────────────────────────────────────────────────

def get_semantic_vector_with_context(
    transcription:  str,
    kg_context_str: str  = "",
    scenario:       str  = "",
    partner:        str  = "",
    model_name:     str  = LLM_MODEL,
    embed_model:    str  = EMBED_MODEL,
    ollama_chat_url: str = OLLAMA_CHAT_URL,
    ollama_embed_url: str = OLLAMA_EMBED_URL,
    temperature:    float = 0.3,
) -> np.ndarray:
    """
    Full context-engineered pipeline:
      1. Build grounded prompt
      2. Get emotional keywords from LLM (context-aware)
      3. Embed keywords with nomic-embed-text
      4. Return 768-d float32 array

    This is a drop-in replacement for the original get_semantic_vector() +
    get_llm_reasoning() pair in benchmark_multimodal.py / rehabilitate_speech.py.
    """
    reasoning = get_llm_reasoning_with_context(
        transcription=transcription,
        kg_context_str=kg_context_str,
        scenario=scenario,
        partner=partner,
        model_name=model_name,
        ollama_url=ollama_chat_url,
        temperature=temperature,
    )

    try:
        res = requests.post(
            ollama_embed_url,
            json={"model": embed_model, "input": reasoning},
            timeout=30,
        )
        res.raise_for_status()
        embedding = res.json()["embeddings"][0]
        return np.array(embedding, dtype=np.float32)
    except Exception as e:
        print(f"[ContextEngineering] Embedding error: {e} — returning zero vector.")
        return np.zeros(768, dtype=np.float32)


# ── STANDALONE TEST ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print(" Context Engineering Module — Standalone Test")
    print("=" * 70)

    # Simulate a KG context string (normally produced by format_context_for_prompt)
    mock_kg = """
Common phrases  : I am ready, more juice, my leg hurts
Preferred vocabulary for this context:
  • 'hungry' → Aarav says: "want food,tummy hurts"  [high frequency]
  • 'juice'  → Aarav says: "juice"                  [high frequency]
  • 'pain'   → Aarav says: "my leg hurts"            [medium frequency]
Past successful utterances in similar contexts:
  • "more juice please, I am ready mama"  (MCDS: 0.61)
"""

    test_cases = [
        {"transcript": "hurt leg bad",   "scenario": "morning",  "partner": "Priya"},
        {"transcript": "play want now",  "scenario": "school",   "partner": "Rohan"},
        {"transcript": "tired stop",     "scenario": "therapy",  "partner": "Dr. Meera"},
    ]

    for tc in test_cases:
        print(f"\n  Transcript : \"{tc['transcript']}\"")
        print(f"  Scenario   : {tc['scenario']}, Partner: {tc['partner']}")

        prompt = build_context_prompt(
            transcription=tc["transcript"],
            kg_context_str=mock_kg,
            scenario=tc["scenario"],
            partner=tc["partner"],
        )

        print("\n  ── Generated Prompt (preview) ─────────────────────────")
        preview_lines = prompt.splitlines()
        for line in preview_lines[:25]:
            print(f"    {line}")
        if len(preview_lines) > 25:
            print(f"    ... ({len(preview_lines) - 25} more lines)")
        print()

    print("[OK] context_engineering.py loaded successfully.")
    print("     To test LLM calls, ensure 'ollama serve' is running.")

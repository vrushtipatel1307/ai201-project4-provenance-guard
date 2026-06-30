"""Detection signals for Provenance Guard.

Milestone 4 implements:
  - Signal 1: LLM-based classification via Groq
  - Signal 2: Stylometric heuristics (pure Python)
  - Confidence scoring: weighted combination of both signals
"""

import json
import os
import re
from statistics import variance

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Model is fixed by the project spec.
_MODEL = "llama-3.3-70b-versatile"

# Thresholds from planning.md section 2. A score is the estimated
# probability that the text was AI-generated.
HUMAN_CEILING = 0.40   # below this -> likely human
AI_FLOOR = 0.70        # at/above this -> likely AI
# Between the two -> uncertain. The uncertain band is intentionally wide
# because a false positive (calling a human's work AI) is the costly error.

_client = None


def _get_client():
    """Lazily build the Groq client so importing this module never crashes
    when the key is missing (e.g. during unrelated unit tests)."""
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your .env file."
            )
        _client = Groq(api_key=api_key)
    return _client


_SYSTEM_PROMPT = (
    "You are a forensic text analyst. You judge whether a passage of writing "
    "was most likely produced by an AI language model or written by a human. "
    "You weigh holistic, semantic cues: generic phrasing, even and predictable "
    "rhythm, hedging boilerplate, and a lack of lived specificity all point to "
    "AI; idiosyncratic voice, concrete personal detail, and irregular rhythm "
    "point to a human. You are calibrated and cautious: when evidence is mixed "
    "you return a probability near 0.5 rather than guessing. You always reply "
    "with a single JSON object and nothing else."
)

_USER_TEMPLATE = (
    "Analyze the following text and return JSON with exactly these keys:\n"
    '  "ai_probability": a number from 0.0 (certainly human) to 1.0 '
    "(certainly AI),\n"
    '  "verdict": one of "likely_human", "uncertain", "likely_ai",\n'
    '  "reasoning": one short sentence explaining the main cue you used.\n\n'
    "TEXT:\n"
    '"""\n{text}\n"""'
)


def get_llm_score(text):
    """Signal 1: ask Groq how AI-like the text reads.

    Returns a dict:
        {
          "llm_score": float in [0, 1],   # 1.0 == strongly AI
          "verdict": str,                  # model's own label
          "reasoning": str,                # one-line justification
        }

    Raises RuntimeError if the API call or parsing fails so the caller can
    decide how to surface the error to the user.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("text must be a non-empty string")

    client = _get_client()
    try:
        completion = client.chat.completions.create(
            model=_MODEL,
            temperature=0,  # deterministic-as-possible for a classifier
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _USER_TEMPLATE.format(text=text)},
            ],
        )
        raw = completion.choices[0].message.content
        data = json.loads(raw)
    except Exception as exc:  # network, JSON, or API error
        raise RuntimeError(f"LLM signal failed: {exc}") from exc

    score = data.get("ai_probability")
    try:
        score = float(score)
    except (TypeError, ValueError):
        raise RuntimeError(f"LLM returned a non-numeric ai_probability: {score!r}")

    # Clamp to the valid range — the model occasionally drifts outside [0, 1].
    score = max(0.0, min(1.0, score))

    return {
        "llm_score": round(score, 3),
        "verdict": data.get("verdict", attribution_from_score(score)),
        "reasoning": data.get("reasoning", ""),
    }


def get_stylometric_score(text):
    """Signal 2: Analyze structural properties of the text.

    Computes three metrics and combines them into a single score:
      1. Sentence length variance (low variance = more AI-like)
      2. Type-token ratio (low diversity = more AI-like)
      3. Punctuation density and variety (less variety = more AI-like)

    Returns a dict:
        {
          "stylometric_score": float in [0, 1],  # 1.0 == strongly AI-like
          "metrics": {...}                       # details for debugging
        }
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("text must be a non-empty string")

    # --- Metric 1: Sentence length variance ---
    # Split on sentence boundaries: . ! ?
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) < 2:
        # Too few sentences for variance; treat as neutral
        sentence_var_score = 0.5
    else:
        lengths = [len(s.split()) for s in sentences]
        avg_length = sum(lengths) / len(lengths)
        if avg_length == 0:
            sentence_var_score = 0.5
        else:
            var = variance(lengths) if len(lengths) > 1 else 0
            # Normalize: lower variance (more uniform) -> higher score
            # High variance (irregular) -> lower score
            # Use coefficient of variation for scale-independence
            cv = (var ** 0.5) / avg_length if avg_length > 0 else 0
            # AI text has low CV (~0.3), human has high CV (~0.6)
            # Map: 0.0-0.4 -> 0.0-1.0, clamped
            sentence_var_score = min(1.0, max(0.0, cv / 0.5))

    # --- Metric 2: Type-token ratio (vocabulary diversity) ---
    words = text.lower()
    # Remove punctuation for word tokenization
    words = re.findall(r"\b\w+\b", words)
    if not words:
        ttr_score = 0.5
    else:
        unique_words = len(set(words))
        total_words = len(words)
        ttr = unique_words / total_words if total_words > 0 else 0
        # AI text has lower TTR (~0.45), human has higher TTR (~0.65)
        # Map: lower TTR -> higher score (more AI-like)
        # 0.3-0.7 range maps to 0.0-1.0
        ttr_score = min(1.0, max(0.0, (0.7 - ttr) / 0.4))

    # --- Metric 3: Punctuation density and variety ---
    punct_chars = re.findall(r"[.!?,;:\-'\"]", text)
    punct_count = len(punct_chars)
    unique_punct = len(set(punct_chars))
    total_chars = len(text)
    
    if total_chars == 0:
        punct_score = 0.5
    else:
        # Punctuation density
        density = punct_count / total_chars
        # AI text has lower density (~0.03), human has higher (~0.05)
        # Lower density -> higher score (more AI-like)
        density_norm = min(1.0, max(0.0, (0.07 - density) / 0.04))
        
        # Punctuation variety (higher variety -> more human)
        # Max variety is ~10 unique punct marks
        variety_norm = min(1.0, (10 - unique_punct) / 10) if unique_punct < 10 else 0.0
        
        punct_score = (density_norm + variety_norm) / 2

    # Combine the three metrics (average)
    combined = (sentence_var_score + ttr_score + punct_score) / 3
    stylometric_score = round(combined, 3)

    return {
        "stylometric_score": stylometric_score,
        "metrics": {
            "sentence_variance": round(sentence_var_score, 3),
            "type_token_ratio": round(ttr_score, 3),
            "punctuation": round(punct_score, 3),
        },
    }


def get_linguistic_patterns_score(text):
    """Signal 3: Detect linguistic patterns characteristic of AI text (Stretch Feature).

    Analyzes:
      - Hedging language density (however, moreover, furthermore, etc.)
      - Contraction usage (AI rarely uses contractions)
      - Formal transition phrases
      - Repetitive connectors

    Returns a dict:
        {
          "linguistic_score": float in [0, 1],  # 1.0 == strongly AI-like patterns
          "metrics": {...}                       # details for debugging
        }
    """
    text_lower = text.lower()

    # Common hedging/formal transitions (AI loves these)
    hedging_words = [
        "however", "moreover", "furthermore", "in addition", "notably",
        "it is important to note", "it is essential", "it is worth noting",
        "arguably", "substantially", "significantly", "demonstrably"
    ]
    hedging_count = sum(text_lower.count(word) for word in hedging_words)

    # Contractions (humans use more, AI uses fewer)
    contractions = ["'s", "'t", "'ll", "'ve", "'re", "'d", "'m"]
    contraction_count = sum(text_lower.count(cont) for cont in contractions)

    # Word count for normalization
    words = re.findall(r"\b\w+\b", text)
    word_count = len(words)

    if word_count < 10:
        return {"linguistic_score": 0.5, "metrics": {}}

    # Normalize hedging (higher hedging = higher AI score)
    # Assume 5-10 hedging phrases in 100 words is AI-like
    hedging_density = (hedging_count / word_count) * 100
    hedging_norm = min(1.0, max(0.0, hedging_density / 5))

    # Normalize contractions (lower contractions = higher AI score)
    # Humans use ~1-2 contractions per 100 words
    contraction_density = (contraction_count / word_count) * 100
    # Inverse: low contractions -> high AI score
    contraction_norm = min(1.0, max(0.0, (3 - contraction_density) / 3))

    # Combine
    linguistic_score = (hedging_norm + contraction_norm) / 2
    return {
        "linguistic_score": round(linguistic_score, 3),
        "metrics": {
            "hedging_density": round(hedging_norm, 3),
            "contraction_density": round(contraction_norm, 3),
        },
    }


def combine_signals(llm_score, stylometric_score, linguistic_score=None):
    """Combine signals into a single confidence score (Ensemble approach, Stretch Feature).

    If linguistic_score is provided (Signal 3), uses ensemble voting:
      - 3 signals with equal voting power
      - Final score is average of all three

    Otherwise, uses 2-signal weighting:
      - LLM signal: 60% (captures semantics)
      - Stylometric signal: 40% (captures structure)

    Args:
        llm_score: float in [0, 1]
        stylometric_score: float in [0, 1]
        linguistic_score: float in [0, 1] or None

    Returns:
        combined_score: float in [0, 1], representing P(AI-generated)
    """
    if linguistic_score is not None:
        # Ensemble: 3 equal signals
        combined = (llm_score + stylometric_score + linguistic_score) / 3
    else:
        # 2-signal weighting (fallback)
        combined = 0.6 * llm_score + 0.4 * stylometric_score
    return round(combined, 3)


def attribution_from_score(score):
    """Map a 0-1 AI-probability to one of the three attribution labels.

    Shared by Signal 1 (Milestone 3) and the combined score (Milestone 4)
    so a single set of thresholds governs the whole system.
    """
    if score < HUMAN_CEILING:
        return "likely_human"
    if score >= AI_FLOOR:
        return "likely_ai"
    return "uncertain"


def get_transparency_label(confidence):
    """Generate the exact transparency label text shown to users (Milestone 5).

    Maps confidence score to one of three variants:
      - [0.00, 0.40): high-confidence human
      - [0.40, 0.70): uncertain
      - [0.70, 1.00]: high-confidence AI

    Args:
        confidence: float in [0, 1]

    Returns:
        label_text: str, the exact text to display to a user
    """
    if confidence < HUMAN_CEILING:  # < 0.40
        return (
            "This content shows strong indicators of human authorship. "
            "Our system did not detect significant signs of AI generation."
        )
    elif confidence >= AI_FLOOR:  # >= 0.70
        return (
            "This content shows strong indicators of AI generation. "
            "Our system is fairly confident this was created or substantially "
            "produced by an AI tool."
        )
    else:  # 0.40-0.69: uncertain
        return (
            "We can't confidently determine whether this content was written by a human "
            "or generated by AI. Treat this result as inconclusive."
        )

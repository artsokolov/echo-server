"""
Classifier microservice — hybrid bot/human detector.

Combines:
  1. Zero-shot NLI with semantically-appropriate labels
  2. Heuristic features (LLM opener phrases, informal language, structural patterns)

Exposes POST /predict — returns is_bot_probability in [0, 1].
"""
import logging
import re
from collections import defaultdict
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI
from pydantic import UUID4, BaseModel, StrictStr
from transformers import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "typeform/distilbert-base-uncased-mnli"
MAX_CHARS = 512

_classifier = None
_speaker_history: dict[tuple[str, int], list[str]] = defaultdict(list)
_seen_ids: set[str] = set()

# ── Heuristic feature lists ────────────────────────────────────────────────────

# Classic LLM/chatbot opener patterns — very strong bot signals
BOT_PHRASES = [
    "of course", "certainly!", "certainly,", "absolutely!", "absolutely,",
    "great question", "that's a great", "i'd be happy to", "i'm happy to",
    "i would be happy", "i'd love to", "i'm here to help",
    "as an ai", "as a language model", "as an assistant",
    "i understand your", "i can help you", "let me help",
    "please feel free", "please let me know", "feel free to ask",
    "thank you for", "i appreciate your", "i hope this helps",
    "in conclusion", "in summary", "to summarize",
    "additionally,", "furthermore,", "moreover,", "however,",
    "it's important to note", "it is worth noting", "please note that",
    "i apologize", "i'm sorry to hear", "i understand that",
    "allow me to", "i'll do my best",
]

# Informal human markers — strong human signals
# Words checked as whole tokens (split-based) to avoid false substring matches
HUMAN_INFORMAL_WORDS = {
    "lol", "lmao", "lmfao", "haha", "hehe", "hahaha", "omg", "omfg",
    "wtf", "idk", "tbh", "btw", "ngl", "imo", "imho", "irl",
    "bruh", "bro", "dude", "mate", "yo", "hey", "sup", "wassup",
    "yeah", "yep", "nah", "nope", "yup", "yolo", "swag",
    "gonna", "wanna", "gotta", "kinda", "sorta", "dunno", "lemme", "gimme",
    "thx", "ty", "np", "ikr", "smh", "fwiw", "afk", "gg", "rn",
    "hmm", "ugh", "meh", "welp", "aight", "alright",
    "u", "ur", "cuz", "coz", "tho", "tbf", "istg", "imo",
    "dope", "sick", "lit", "vibe", "vibes", "lowkey", "highkey",
    "fr", "frfr", "nah", "bro", "slay", "fam", "no cap", "cap",
}
# Substrings that are safe to match (longer, no false positive risk)
HUMAN_INFORMAL_SUBSTRINGS = ["...", "hm,", "eh,", "man,", "r u ", "lmfao"]

# Formal connectives used heavily by LLMs
BOT_CONNECTIVES = [
    "additionally", "furthermore", "moreover", "nevertheless",
    "in conclusion", "in summary", "to summarize", "consequently",
    "therefore", "thus,", "hence,",
]

# ── ML model templates ─────────────────────────────────────────────────────────

TEMPLATE_CONFIGS = [
    (["AI-generated", "human-written"], "This text is {}."),
    (["automated chatbot", "real human"], "This message was sent by a {}."),
    (["scripted computer program", "genuine person"], "This was written by a {}."),
]


class IncomingMessage(BaseModel):
    text: StrictStr
    dialog_id: UUID4
    id: UUID4
    participant_index: int


class Prediction(BaseModel):
    id: UUID4
    message_id: UUID4
    dialog_id: UUID4
    participant_index: int
    is_bot_probability: float


def load_model():
    global _classifier
    if _classifier is None:
        logger.info("Loading classifier model: %s", MODEL_NAME)
        _classifier = pipeline(
            "zero-shot-classification",
            model=MODEL_NAME,
            device=-1,
        )
        logger.info("Classifier ready.")
    return _classifier


def ml_score(text: str) -> float:
    """Average bot probability across multiple NLI templates."""
    clf = load_model()
    scores = []
    for (bot_label, human_label), template in TEMPLATE_CONFIGS:
        result = clf(
            text,
            candidate_labels=[bot_label, human_label],
            hypothesis_template=template,
        )
        bot_idx = result["labels"].index(bot_label)
        scores.append(result["scores"][bot_idx])
    return sum(scores) / len(scores)


def heuristic_score(texts: list[str]) -> float:
    """Rule-based bot probability from text features."""
    combined = " ".join(texts).lower()
    # Normalize: strip punctuation from word tokens for reliable matching
    raw_words = combined.split()
    words = [w.strip(".,!?;:\"'()[]") for w in raw_words]
    n_words = max(len(words), 1)

    score = 0.5

    # Strong bot signals: LLM opener phrases
    bot_phrase_hits = sum(phrase in combined for phrase in BOT_PHRASES)
    score += min(bot_phrase_hits * 0.16, 0.45)

    # Formal connectives typical of LLMs
    connective_hits = sum(w in combined for w in BOT_CONNECTIVES)
    score += min(connective_hits * 0.06, 0.15)

    # Human signals: informal markers (word-level to avoid false positives)
    informal_hits = sum(w in HUMAN_INFORMAL_WORDS for w in words)
    informal_hits += sum(marker in combined for marker in HUMAN_INFORMAL_SUBSTRINGS)
    score -= min(informal_hits * 0.10, 0.35)

    # Personal first-person pronouns → human signal
    pronouns = [w for w in words if w in ("i", "me", "my", "mine", "myself")]
    pronoun_ratio = len(pronouns) / n_words
    score -= min(pronoun_ratio * 3.0, 0.20)

    # Contractions → human signal (bots often avoid them in formal mode)
    contractions = re.findall(
        r"(don't|can't|won't|i'm|i've|i'll|it's|that's|didn't|doesn't|"
        r"isn't|you're|we're|they're|i'd|couldn't|wouldn't|shouldn't|"
        r"what's|where's|how's|who's|there's|here's|let's|wasn't|weren't|"
        r"haven't|hasn't|ain't)",
        combined,
    )
    score -= min(len(contractions) * 0.06, 0.18)

    # Very long average message → bot signal (LLMs write long responses)
    avg_msg_len = n_words / max(len(texts), 1)
    if avg_msg_len > 40:
        score += 0.12
    elif avg_msg_len > 25:
        score += 0.06

    # Very short messages are ambiguous — slight human lean
    if avg_msg_len < 4:
        score -= 0.05

    # Ends messages with "?" frequently → human signal (genuine curiosity)
    question_ratio = sum(t.strip().endswith("?") for t in texts) / max(len(texts), 1)
    score -= min(question_ratio * 0.10, 0.10)

    # Exclamation marks ratio — moderate use is human, excessive is bot
    excl_count = combined.count("!")
    if excl_count > 3:
        score += 0.05  # bots sometimes over-use exclamation marks

    return max(0.02, min(0.98, score))


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Classifier Service", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(msg: IncomingMessage) -> Prediction:
    msg_id = str(msg.id)
    if msg_id not in _seen_ids:
        _seen_ids.add(msg_id)
        key = (str(msg.dialog_id), msg.participant_index)
        _speaker_history[key].append(msg.text)

    texts = _speaker_history[(str(msg.dialog_id), msg.participant_index)]
    combined = " ".join(texts)[:MAX_CHARS]

    h_score = heuristic_score(texts)
    m_score = ml_score(combined)

    # If heuristics are confident, trust them almost fully
    if h_score >= 0.72 or h_score <= 0.22:
        probability = float(max(0.02, min(0.98, 0.88 * h_score + 0.12 * m_score)))
    else:
        # Ambiguous — blend heuristics with ML
        probability = float(max(0.02, min(0.98, 0.70 * h_score + 0.30 * m_score)))

    logger.info(
        "dialog=%s heuristic=%.3f ml=%.3f final=%.3f",
        msg.dialog_id, h_score, m_score, probability,
    )
    return Prediction(
        id=uuid4(),
        message_id=msg.id,
        dialog_id=msg.dialog_id,
        participant_index=msg.participant_index,
        is_bot_probability=probability,
    )

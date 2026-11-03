"""Single entry point to the handwriting vision model.

read_answer() sends the crop to qwen2.5vl:7b on a local Ollama server
(no Tesseract or other traditional OCR engines).

For multiple-choice answers (options=LETTER_OPTIONS or ROMAN_OPTIONS) it also
reads the option label the answer starts with and sends the answer to review when that can't be trusted:
no letter at the start, or several options written (the model copies
crossed-out options instead of leaving them out).
"""
import base64
import json
import math
import os
import re
import urllib.request
from typing import Optional, TypedDict

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
# 7b vs 3b on 47 real crops + 20 synthetic: 55 vs 51 right (ignoring spaces), multiple-choice
# option right 15/18 vs 14/18, about 5.5 s vs 4.4 s per answer once loaded (RTX 3050 6 GB).
MODEL = "qwen2.5vl:7b"
# Compared with the one-line prompt on 29 real crops + 20 synthetic: exact 15/29 vs 13/29
# real, 20/20 vs 19/20 synthetic. Saying "lowercase" for option labels made the model
# lowercase everything and invent labels, hence the "only if written" wording.
PROMPT = (
    "Transcribe the handwritten answer exactly as written, character for character.\n"
    "- Copy spelling, capitalisation and punctuation exactly as the writer did, even when wrong. "
    "Never correct anything.\n"
    "- Only if the writer wrote an option label such as (ii) or (iv), type it with the ordinary "
    "keyboard letters i, v and x, not special numeral symbols. Never add a label that is not written.\n"
    "- Type fractions as a/b and mixed numbers as 1 2/3. Plain text only, no LaTeX.\n"
    "- Leave out anything crossed out.\n"
    "Return only the text. If it is unreadable, return an empty string."
)
# The model still emits Unicode numeral symbols (Ⅲ) at times; students write option labels
# in lowercase letters, so map both cases to those.
ROMAN_NUMERALS = str.maketrans({
    chr(base + i): s
    for base in (0x2160, 0x2170)
    for i, s in enumerate(["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii"])
})
# Placeholder until a threshold is chosen from score.py's sweep on real labelled crops.
REVIEW_THRESHOLD = 0.5

LETTER_OPTIONS = ("A", "B", "C", "D")
ROMAN_OPTIONS = ("i", "ii", "iii", "iv")
# Cyrillic and Greek letters the model sometimes emits for a handwritten option letter.
LOOKALIKES = {"А": "A", "а": "A", "В": "B", "в": "B", "С": "C", "с": "C",
              "Α": "A", "α": "A", "Β": "B", "β": "B"}
# An option label at the start of the answer or after a "/" (how students separate a
# rewritten choice); "(A)" inside a sentence such as "Both (A) and (R)" is not one.
OPTION_LABEL = re.compile(r"(?:^|/)\s*\(?([A-Za-z]{1,4})\)")


class Answer(TypedDict):
    text: Optional[str]
    confidence: Optional[float]       # minimum token probability: the review signal
    confidence_mean: Optional[float]  # geometric mean token probability, kept for comparison
    needs_review: bool
    option: Optional[str]             # multiple choice only: the label, None when not trustworthy
    review_reasons: list[str]
    tokens: list[tuple[str, float]]   # (token, logprob) as the model wrote them, for finer confidence


def check_option(text: str, options: tuple[str, ...]) -> tuple[str, Optional[str], list[str]]:
    """Normalize a look-alike leading letter; return (text, option, review reasons)."""
    if text and text[0] in LOOKALIKES:
        text = LOOKALIKES[text[0]] + text[1:]
    by_key = {o.lower(): o for o in options}
    labels = [by_key.get(m.lower()) for m in OPTION_LABEL.findall(text)]
    starts = text.lstrip("( ").lower()
    if not labels or labels[0] is None or not starts.startswith(labels[0].lower() + ")"):
        return text, None, ["no option label at the start"]
    if len(labels) > 1:
        return text, None, [f"several options written ({', '.join(l or '?' for l in labels)}), "
                            "maybe crossed out"]
    return text, labels[0], []


def read_answer(image_path: str, question_id: str,
                options: Optional[tuple[str, ...]] = None) -> Answer:
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("ascii")

    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT, "images": [image_b64]}],
        "stream": False,
        "logprobs": True,
        "options": {"temperature": 0},
    }
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.load(resp)

    text = data["message"]["content"].strip().translate(ROMAN_NUMERALS)
    tokens = [(t.get("token", ""), t["logprob"]) for t in data.get("logprobs") or [] if t.get("logprob") is not None]
    logprobs = [lp for _, lp in tokens]
    confidence = math.exp(min(logprobs)) if logprobs else None
    confidence_mean = math.exp(sum(logprobs) / len(logprobs)) if logprobs else None
    reasons = []
    if not text:
        reasons.append("no text")
    # Unknown confidence can't clear the threshold, so it is flagged too.
    if confidence is None or confidence < REVIEW_THRESHOLD:
        reasons.append("low confidence")
    option = None
    if options:
        text, option, option_reasons = check_option(text, options)
        reasons += option_reasons
    return {"text": text, "confidence": confidence, "confidence_mean": confidence_mean,
            "needs_review": bool(reasons), "option": option, "review_reasons": reasons, "tokens": tokens}

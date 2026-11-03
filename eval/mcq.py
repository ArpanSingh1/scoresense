"""Check a booklet's multiple-choice answers against the exam's answer key.

For each MCQ question the model reads the question's first crop. The question number the student
wrote ("Q1.", "Ans 1.") is removed, and the option label is taken by model.check_option's rules:
an answer with no option label at the start, or with several options (a crossed-out first
choice), goes to the teacher instead of being marked. Options written (a)-(d) or (i)-(iv) count
as A-D.

    right   the option matches the key: the full marks are suggested
    wrong   a clear option that differs from the key: 0 is suggested, and the teacher is asked to look
    review  the option can't be trusted: no mark is suggested
"""
import math
import re
from pathlib import Path
from typing import Callable, Optional

from model import LETTER_OPTIONS, REVIEW_THRESHOLD, ROMAN_OPTIONS, check_option, read_answer

# An option label, as written by students: (C), C), (c), (ii), ii).
OPTION_START = re.compile(r"\(?\s*(?:[A-Da-d]|i{1,3}|iv)\s*\)")
ROMAN_TO_LETTER = dict(zip(ROMAN_OPTIONS, LETTER_OPTIONS))
# What students write before the answer: "Q2.", "Ans 2.", read at times as "02.", "Aw10.", "①5."
NUMBER_WORDS = re.compile(r"(?i)\b(?:q|ans|aus|ams|aw|an|o)\b")


def only_number(text: str) -> bool:
    """True when text is nothing but a question number such as "Q15." or "Ans 2.\n02."."""
    rest = re.sub(r"[\d\W_\u2460-\u2473]", " ", text)
    return not NUMBER_WORDS.sub(" ", rest).strip()


def strip_number(text: str) -> str:
    """The answer without the question number in front: "Q1. Ans 1. (C) 1:4" -> "(C) 1:4"."""
    m = OPTION_START.search(text)
    if m and only_number(text[:m.start()]):
        return text[m.start():]
    return text


def label_confidence(tokens: list[tuple[str, float]], label: str) -> Optional[float]:
    """Probability of the option label itself ("C", or "ii" over one or more tokens), the first
    time the model wrote it; None if it can't be found among the tokens."""
    want = label.lower()
    for i in range(len(tokens)):
        got, lps = "", []
        for tok, lp in tokens[i:i + 3]:
            letters = re.sub(r"[^a-z]", "", tok.lower())
            if not letters:
                if got:
                    break
                continue
            got += letters
            lps.append(lp)
            if not want.startswith(got):
                break
            if got == want:
                return math.exp(min(lps))
    return None


def read_choice(image: Path) -> dict:
    """{"read": text, "option": "A"-"D" or None, "reasons": [...]}

    The answer's overall confidence is low whenever a long option sentence has an unclear word,
    so for MCQs only the confidence of the option label counts (as for question numbers)."""
    ans = read_answer(str(image), "")
    text = strip_number(ans["text"] or "")
    if not text.strip() or only_number(text):
        return {"read": text, "option": None, "reasons": ["nothing written"]}
    reasons = [r for r in ans["review_reasons"] if r not in ("no text", "low confidence")]
    _, option, why = check_option(text, LETTER_OPTIONS)
    label = option
    if option is None and why == ["no option label at the start"]:
        _, roman, why_roman = check_option(text, ROMAN_OPTIONS)
        if roman or why_roman != why:
            option, why, label = ROMAN_TO_LETTER.get(roman), why_roman, roman
    if label and not why:
        conf = label_confidence(ans.get("tokens", []), label)
        if conf is None or conf < REVIEW_THRESHOLD:
            reasons.append("option letter not clear" if conf is not None else "low confidence")
    elif "low confidence" in ans["review_reasons"]:
        reasons.append("low confidence")
    return {"read": text, "option": option if not why and not reasons else None, "reasons": reasons + why}


def read_question(crops_dir: Path, crops: list[dict]) -> dict:
    """Read every crop of the question (up to 4) and decide from all of them.

    The first crop is often just the "Q15." line, with the answer in the next one or on the next
    page. A student may also cross out an answer and write it again further on; the model does not
    reliably leave crossed-out writing out, so two different options anywhere means the teacher
    decides."""
    found = []
    for c in crops[:4]:
        r = read_choice(crops_dir / c["file"])
        if r["reasons"] != ["nothing written"]:
            found.append((c["page"], r))
    if not found:
        return {"read": "", "option": None, "reasons": ["nothing written"]}
    options = {r["option"] for _, r in found if r["option"]}
    read = " / ".join(r["read"] for _, r in found)
    if len(options) > 1:
        where = ", ".join(f"{r['option']} on page {p}" for p, r in found if r["option"])
        return {"read": read, "option": None,
                "reasons": [f"different options written ({where}); one may be crossed out"]}
    reasons = [x for _, r in found for x in r["reasons"]]
    return {"read": read, "option": next(iter(options)) if options and not reasons else None,
            "reasons": list(dict.fromkeys(reasons))}


def check(result: dict, crops_dir: Path, mcq: list[int], key: dict[str, str],
          progress: Callable[[str], None] = print) -> dict:
    """Add an "mcq" entry to each keyed MCQ question in result (in place) and return it."""
    todo = [q for q in result["questions"]
            if q["number"] in key and int(q["number"].split(".")[0]) in mcq and q["crops"]]
    for i, q in enumerate(todo, 1):
        progress(f"Checking multiple-choice answers: {i} of {len(todo)}")
        r = read_question(crops_dir, q["crops"])
        expected = key[q["number"]]
        r["key"] = expected
        r["verdict"] = "review" if r["option"] is None else "right" if r["option"] == expected else "wrong"
        q["mcq"] = r
    return result

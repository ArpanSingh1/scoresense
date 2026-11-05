"""Check a transcribed long answer against the teacher's marking points.

EXPERIMENTAL, not used by mark.py. Tested 2026-11-03 on one real 3-paragraph answer with
qwen2.5vl:7b as the judge: asked for quoted evidence, it copied the marking points back as
"evidence" (rejected by evidence_found below); asked to pick a sentence number, it found every
real point but also gave credit to unrelated sentences; asked to verify yes/no, it rejected 4 of
5 correct matches. Needs a stronger text model as judge, and single-idea marking points.

check_points() asks the model (text only, the same one that reads the crops)
which of the key points the answer makes, with the student's own words as
evidence for each. A point counts only when its evidence is actually found in
the answer, so the model can't give credit for something the student didn't
write. The result is a suggestion: the teacher confirms every long answer.

    {"type": "points", "marks": 5, "needed": 5,
     "question": "Explain how cultural processes created a sense of collective belonging.",
     "points": ["Bharat Mata image by Bankim Chandra Chattopadhyay / Abanindranath Tagore", ...]}

"needed" (default: all points) is how many points earn full marks, for
"any three" questions; marks are suggested in proportion, in half marks.
"""
import json
import re
import urllib.request
from typing import Optional, TypedDict

from model import MODEL, OLLAMA_URL

PROMPT = """You are helping a teacher mark a student's exam answer.

Question: {question}

Marking points:
{points}

Student's answer (transcribed from handwriting, so ignore spelling mistakes):
\"\"\"
{answer}
\"\"\"

For each marking point, decide whether the student's answer makes that point. The idea must be
clearly stated in the answer; different words are fine, but do not give credit for something the
answer does not say. For a covered point, copy the exact words from the student's answer that make
it as evidence (one short phrase or sentence). For a point that is not covered, leave evidence empty."""

SCHEMA = {
    "type": "object",
    "properties": {
        "points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "number": {"type": "integer"},
                    "covered": {"type": "boolean"},
                    "evidence": {"type": "string"},
                },
                "required": ["number", "covered", "evidence"],
            },
        }
    },
    "required": ["points"],
}
# Share of the evidence's words that must appear in the answer for it to count as a quote.
MIN_EVIDENCE_MATCH = 0.8


class PointResult(TypedDict):
    point: str
    covered: bool       # covered and the evidence was found in the answer
    evidence: str
    note: str           # why a claimed point was not counted, if it wasn't


class PointsCheck(TypedDict):
    points: list[PointResult]
    covered: int
    suggested_marks: float


def words(s: str) -> list[str]:
    """Latin and Devanagari words (the vowel signs of Hindi are part of the word)."""
    return re.findall(r"[a-z0-9ऀ-ॣ०-ॿ]+", s.lower())


def evidence_found(evidence: str, answer: str) -> bool:
    """The quote's words are in the answer. Two words at least, so a lone common word isn't a quote,
    unless the answer itself is that short ("सरल वाक्य", "कर्तृवाच्य")."""
    ev, ans = words(evidence), words(answer)
    least = 1 if len(ans) <= 3 else 2
    return len(ev) >= least and sum(w in set(ans) for w in ev) / len(ev) >= MIN_EVIDENCE_MATCH


def suggest(covered: int, needed: int, marks: float) -> float:
    return round(min(covered, needed) / needed * marks * 2) / 2


def check_points(answer: str, question: str, points: list[str], marks: float,
                 needed: Optional[int] = None) -> PointsCheck:
    needed = needed or len(points)
    prompt = PROMPT.format(question=question, answer=answer,
                           points="\n".join(f"{i}. {p}" for i, p in enumerate(points, 1)))
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "stream": False, "format": SCHEMA, "options": {"temperature": 0}}
    req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        claimed = {p["number"]: p for p in json.loads(json.load(resp)["message"]["content"])["points"]}

    results = []
    for i, point in enumerate(points, 1):
        c = claimed.get(i, {"covered": False, "evidence": ""})
        evidence = c["evidence"].strip()
        ok = c["covered"] and evidence_found(evidence, answer)
        note = ""
        if c["covered"] and not ok:
            note = "model claimed it, but its evidence is not in the answer"
        results.append({"point": point, "covered": ok, "evidence": evidence if ok else "", "note": note})
    covered = sum(r["covered"] for r in results)
    return {"points": results, "covered": covered, "suggested_marks": suggest(covered, needed, marks)}

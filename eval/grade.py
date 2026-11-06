"""Grade long answers against the teacher's marking points.

For each question that has marking points (test["points"]), the answer is transcribed from its crops
(model.read_answer, the reading model) and then a judge model is asked about each point separately:
is the idea in the answer, and which of the student's own words make it. A point counts only when that
quote is really in the transcript (points.evidence_found), so the judge can't give credit for words the
student didn't write. Marks are suggested in proportion to the points made, in half marks, and the
teacher confirms them.

Judges compared with bench_judge.py on a real answer, a weak one with wrong facts, and a paraphrased
one (22 points): qwen3-vl:8b-instruct 22/22 with no false credit, qwen2.5vl:7b 20/22 with none. All
transcripts are read first and judged after, so each model is loaded once per sheet.
"""
import json
import re
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Optional

from model import OLLAMA_URL, read_answer
from parts import LEAD_NUMBER
from points import evidence_found, words

JUDGE_MODEL = "qwen3-vl:8b-instruct"
PROMPT = """You are helping a teacher mark a student's exam answer.

Question {number}. Marks: {marks:g}

Student's answer (transcribed from handwriting, so ignore spelling mistakes):
\"\"\"
{answer}
\"\"\"

Marking point: {point}

Does the student's answer clearly make this point? Different wording is fine, but the idea itself must be
in the answer; words that merely appear in it are not enough, and a wrong fact does not make the point.
If it does, copy the student's exact words that make the point (one sentence or phrase) into "quote".
If not, set "made" to false and "quote" to "".
"""
DIAGRAM = re.compile(r"(?i)\b(diagrams?|draw[ns]?|drawing|sketch|labelled|graph)\b")
SCHEMA = {"type": "object", "properties": {"made": {"type": "boolean"}, "quote": {"type": "string"}},
          "required": ["made", "quote"]}
# Grammar terms spelt almost alike: a misread term must still be nearest the right one, so "कर्मवाद्य"
# counts as कर्मवाच्य, but a wrong कर्मवाच्य never counts as कर्तृवाच्य.
TERM_FAMILIES = [
    ["कर्तृवाच्य", "कर्मवाच्य", "भाववाच्य"],
    ["सरल", "संयुक्त", "मिश्र"],
    ["संज्ञा", "सर्वनाम", "विशेषण", "क्रियाविशेषण", "क्रिया", "अव्यय"],
    ["उपमा", "रूपक", "उत्प्रेक्षा", "मानवीकरण", "अतिशयोक्ति", "श्लेष", "यमक", "अनुप्रास"],
    ["जातिवाचक", "व्यक्तिवाचक", "भाववाचक", "समूहवाचक", "द्रव्यवाचक"],
    ["पुल्लिंग", "स्त्रीलिंग"], ["एकवचन", "बहुवचन"],
]
AUXILIARIES = {"था", "थी", "थे", "है", "हैं"}      # "आया गया" for "आया गया था" is still right


def _plain(text: str) -> str:
    return " ".join(re.findall(r"[ऀ-ॣ०-ॿ\w]+", re.sub(r"\([^)]*\)", " ", text)))


def close_match(point: str, transcript: str, at: float = 0.8) -> str:
    """The student's words that give a short point letter by letter, allowing for reading errors
    ("नेलाजी ... चर्म" for नेताजी ... चश्मा), or "" when it isn't there.

    The point's label ("कर्मवाच्य: ...") is left out. Every word of the point must be there except an
    auxiliary, so a sentence still in the active voice (no "द्वारा") is not taken. One letter that changes
    the case (उसने for उससे) can't be told from a misread: the note on the point tells the teacher."""
    want = _plain(point.split(":", 1)[-1])
    got = _plain(transcript).split()
    n = len(want.split())
    if not want or not got or n > 14 or len(got) > 24:
        return ""
    best, span = 0.0, ""
    for size in {max(1, n - 1), n, n + 1}:
        for i in range(max(1, len(got) - size + 1)):
            cand = " ".join(got[i:i + size])
            r = SequenceMatcher(None, cand, want).ratio()
            if r > best:
                best, span = r, cand
    if best < at:
        return ""
    near = lambda a, b: SequenceMatcher(None, a, b).ratio()
    if any(max(near(w, x) for x in span.split()) < 0.5 for w in want.split() if w not in AUXILIARIES):
        return ""
    for family in TERM_FAMILIES:
        for term in set(family) & set(want.split()):
            close = [w for w in span.split() if near(w, term) >= 0.6]
            if all(max(family, key=lambda t: near(w, t)) != term for w in close):
                return ""
    return span


def judge(answer: str, point: str, number: str, marks: float) -> dict:
    body = {"model": JUDGE_MODEL, "stream": False, "format": SCHEMA,
            "messages": [{"role": "user", "content": PROMPT.format(number=number, marks=marks, answer=answer, point=point)}],
            "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 600}}
    req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        content = json.load(resp)["message"]["content"]
    try:
        out = json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.S)
        out = json.loads(m.group(0)) if m else {}
    said = bool(out.get("made"))
    quote = str(out.get("quote", "")).strip()
    made = said and evidence_found(quote, answer)
    note = "the judge's quote is not in the answer" if said and not made else ""
    return {"point": point, "made": made, "quote": quote if made else "", "note": note}


ALTERNATIVES = re.compile(r"\s*;\s*(?:अथवा|या फिर|or)\s+", re.I)
# A point that is a slot for any correct point: "एक अन्य सही बिंदु: पुल्लिंग / बहुवचन", "कोई दूसरा तथ्य: ...".
OTHER_SLOT = re.compile(r"अन्य\s+सही\s+बिंदु|कोई\s+दूसरा")


def sides(point: str) -> list[str]:
    """The alternatives a point accepts, each with the point's label: "असंभव नहीं, क्योंकि ...; अथवा असंभव
    है, ...", or short options split by "/" ("एक अन्य सही बिंदु: पुल्लिंग / बहुवचन / कर्म कारक")."""
    label, _, body = point.partition(":") if ":" in point.split(";")[0] else ("", "", point)
    parts = ALTERNATIVES.split(body)
    if len(parts) < 2:
        # Only " / " with spaces: "1/R = 1/10", "eggs/ova" and "पाते/सकते" are one point.
        options = [o.strip() for o in re.split(r"\s+/\s+", body)]
        ok = lambda o: 0 < len(o.split()) <= 8 and o.count("(") == o.count(")") and not re.search(r"\d", o)
        parts = options if len(options) > 1 and all(map(ok, options)) else [body]
        # Another ending of the same sentence ("दादाजी अब चल नहीं पाते / नहीं सकते"): the whole sentence.
        for i in range(1, len(parts)):
            first, before = parts[i].split()[0], parts[0].split()
            at = max((k for k, w in enumerate(before) if w == first and k >= len(before) - 3), default=None)
            if at is not None and len(parts[i].split()) < len(before):
                parts[i] = " ".join(before[:at] + parts[i].split())
    if len(parts) < 2:
        return [point]
    return [f"{label}: {s.strip()}" if label else s.strip() for s in parts]


def judge_any(answer: str, point: str, number: str, marks: float) -> dict:
    """A point that accepts any of its sides: each is judged alone, since the judge asked about all of them
    at once wants all of them."""
    options = sides(point)
    if len(options) < 2:
        return judge(answer, point, number, marks)
    for side in options:
        c = judge(answer, side, number, marks)
        if c["made"]:
            return {**c, "point": point}
    return {**c, "point": point}


OTHER_PROMPT = """You are helping a teacher mark a student's exam answer.

Question {number}: {question} Marks: {marks:g}

Student's answer (transcribed from handwriting, so ignore spelling mistakes):
\"\"\"
{answer}
\"\"\"

The marking scheme lists these points and also gives credit for "any other relevant point":
{listed}

List the points the student makes that are NOT among the listed ones but are themselves a correct and
relevant answer to the question. A wrong fact, a point off the question, or a repeat of the question is not
a point. For each, give a short statement of the point and copy the student's exact words that make it
into "quote". If there are none, give an empty list.
"""
OTHER_SCHEMA = {"type": "object", "properties": {"points": {"type": "array", "items": {
    "type": "object", "properties": {"point": {"type": "string"}, "quote": {"type": "string"}},
    "required": ["point", "quote"]}}}, "required": ["points"]}
OTHER_NOTE = "not in the marking scheme, which accepts \"any other relevant point\": check it"


def judge_other(answer: str, question: str, listed: list[str], number: str, marks: float, used: list[str],
                most: int) -> list[dict]:
    """Up to `most` correct points beyond the scheme's list, for a scheme that accepts "Any other relevant
    point" (Rio Summit outcomes: "Agenda 21 was adopted"). Each needs a quote that is in the answer and
    not already the quote of a point given; the teacher is told to check them."""
    body = {"model": JUDGE_MODEL, "stream": False, "format": OTHER_SCHEMA,
            "messages": [{"role": "user", "content": OTHER_PROMPT.format(
                number=number, question=question or "(see the marking points)", marks=marks, answer=answer,
                listed="\n".join(f"- {p}" for p in listed))}],
            "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 800}}
    req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        content = json.load(resp)["message"]["content"]
    try:
        found = json.loads(content).get("points", [])
    except (json.JSONDecodeError, AttributeError):
        found = []
    return other_points(found, answer, used, most)


def other_points(found: list, answer: str, used: list[str], most: int) -> list[dict]:
    """judge_other's check on what the judge gave: real quotes only, none overlapping a quote used."""
    out, taken = [], list(used)
    for f in found if isinstance(found, list) else []:
        quote = str((f or {}).get("quote", "")).strip() if isinstance(f, dict) else ""
        point = str(f.get("point", "")).strip() if isinstance(f, dict) else ""
        if len(out) >= most or not quote or not point or not evidence_found(quote, answer):
            continue
        if any(SequenceMatcher(None, quote.lower(), t.lower()).find_longest_match().size >= 0.6 * min(len(quote), len(t))
               for t in taken if t):
            continue                                    # the same words as a point already given
        taken.append(quote)
        out.append({"point": f"Other relevant point: {point}", "made": True, "quote": quote,
                    "note": OTHER_NOTE, "other": True})
    return out


def suggest(made: int, needed: int, marks: float) -> float:
    return round(min(made, needed) / needed * marks * 2) / 2


def split_by_text(questions: list[dict], q: dict, texts: list[str], points: dict[str, dict]) -> None:
    """The next question inside this answer, its number missed in the margin (27's answer going on with
    "28. Three characteristics ..."): from that line on it becomes the next question, flagged (in place).
    texts: what was read from each of q's crops. Only for a next number above 9 that is nowhere else on
    the sheet and has marking points, so an answer's own "3." never cuts it."""
    if not q["number"].isdigit():
        return
    nxt = str(int(q["number"]) + 1)
    if int(nxt) <= 9 or nxt not in points or any(x["number"].split(".")[0] == nxt for x in questions):
        return
    for k, text in enumerate(texts):
        lines = text.splitlines()
        j = next((j for j, line in enumerate(lines) if (k or j) and (m := LEAD_NUMBER.match(line))
                  and m.group(1) == nxt), None)
        if j is None:
            continue
        rest = ["\n".join(lines[j:])] + [t for t in texts[k + 1:] if t]
        new = {"number": nxt, "crops": q["crops"][k:],
               "flags": [f"Q{nxt} was found inside Q{q['number']}'s answer (its number was not read): check it"],
               "transcript": "\n".join(rest)}
        q["transcript"] = "\n".join([t for t in texts[:k] if t] + (["\n".join(lines[:j])] if j else []))
        q["crops"] = q["crops"][:k + (1 if j else 0)]
        questions.insert(questions.index(q) + 1, new)
        return


def grade(result: dict, crops_dir: Path, points: dict[str, dict], scheme: dict[str, float],
          only: Optional[str] = None, progress: Callable[[str], None] = print) -> dict:
    """Add a "grading" entry to each question with marking points (in place) and return result.

    points: {"25": {"points": ["...", ...], "needed": 3}}; only: grade just this question number."""
    pick = lambda: [q for q in result["questions"] if q["number"] in points and points[q["number"]].get("points")
                    and (only is None or q["number"] == only)]
    todo = pick()
    for i, q in enumerate(todo, 1):                     # 1. read every answer (the reading model)
        if q.get("part_of"):
            continue                                    # a part of a question: its own text is read already
        progress(f"Reading long answers: {i} of {len(todo)}")
        texts = [(read_answer(str(crops_dir / c["file"]), q["number"])["text"] or "").strip() for c in q["crops"]]
        q["transcript"] = "\n".join(t for t in texts if t)
        if only is None:
            split_by_text(result["questions"], q, texts, points)
    todo = pick()
    total = sum(len(points[q["number"]]["points"]) for q in todo)
    done = 0
    for q in todo:                                      # 2. judge every point (the judge model)
        spec = points[q["number"]]
        marks = scheme.get(q["number"]) or scheme.get(q["number"].split(".")[0]) or 0
        # Without "Any two points...", a point a mark (36.1: two points printed, 1 mark: either point).
        default = max(1, round(marks)) if marks else len(spec["points"])
        needed = max(1, min(int(spec.get("needed") or default), len(spec["points"])))
        checked = []
        for p in spec["points"]:
            done += 1
            progress(f"Checking marking points: {done} of {total}")
            if not q["transcript"]:
                checked.append({"point": p, "made": False, "quote": "", "note": ""})
            else:
                c = judge_any(q["transcript"], p, q["number"], marks)
                span = "" if c["made"] else next(
                    (s for s in (close_match(side, q["transcript"]) for side in sides(p)) if s), "")
                if span:
                    c.update(made=True, quote=span, note="matched letter by letter: check the writing")
                checked.append(c)
        made = sum(c["made"] for c in checked)
        other = spec.get("other") or any(OTHER_SLOT.search(p) for p in spec["points"])
        if other and q["transcript"] and made < needed:
            progress(f"Checking other relevant points: question {q['number']}")
            checked += judge_other(q["transcript"], spec.get("question", ""), spec["points"], q["number"], marks,
                                   [c["quote"] for c in checked if c["made"]], needed - made)
            made = sum(c["made"] for c in checked)
        # A point about a drawing can't be judged from the transcript: when one is missing and full marks
        # weren't reached in words, the mark is left to the teacher ("diagram"), the suggestion a minimum.
        diagram = made < needed and any(not c["made"] and DIAGRAM.search(c["point"]) for c in checked)
        # A short answer (a grammar term, a rewritten sentence) that misses: one misread word decides it
        # ("कर्तृवाच्य" read as "कर्तव्य"), so the teacher looks at the writing instead of a 0 being suggested.
        short = made < needed and 0 < len(words(q["transcript"])) <= 12
        q["grading"] = {"points": checked, "made": made, "needed": needed, "out_of": marks,
                        "suggested": suggest(made, needed, marks) if marks else None,
                        "empty": not q["transcript"], "diagram": diagram, "short": short}
    return result

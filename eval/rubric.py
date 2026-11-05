"""Make the rubric from the question paper: the MCQ answer key and the marking points of each written answer.

The reading model copies the questions from every page of the paper (read_questions); then the judge model,
which marks the answers later, writes for each question either the correct option or the points a full
answer makes, one fact each (make_rubric). A written answer gets a few more points than its marks, with
"needed" equal to its marks, so a student who makes a different valid point is not marked down.

The model writes this from its own knowledge, which is not always right: the teacher checks the key and
points on the setup page before the answer sheets are marked with them.

    python eval/rubric.py QUESTION_PAPER.pdf "1-20:1, 21-24:2, ..." [mcq numbers as 1-20]
"""
import base64
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from booklet import load_pages
from grade import JUDGE_MODEL
from model import MODEL, OLLAMA_URL

READ_PROMPT = (
    "This is a page of an exam question paper. Copy every question on it, in order:\n"
    "- number: the question number as printed, for example 21 or 34 (for a case-based question, its main number)\n"
    "- text: the question exactly as printed, with its sub-parts; for an internal choice, both questions "
    "with OR between them\n"
    "- options: for a multiple-choice question, the options as printed, for example \"(A) Buddhism\"; "
    "otherwise an empty list\n"
    "Many pages start with instructions or a section heading and have questions below them: skip the "
    "instructions and copy the questions. If a question starts on an earlier page, copy what is on this "
    "page with that question's number. Only if the page has no numbered questions at all (for example a "
    "cover page, or a page in another language than English when the paper has an English version), "
    "return an empty list.")
READ_SCHEMA = {"type": "object", "properties": {"questions": {"type": "array", "items": {
    "type": "object", "properties": {"number": {"type": "integer"}, "text": {"type": "string"},
                                     "options": {"type": "array", "items": {"type": "string"}}},
    "required": ["number", "text", "options"]}}}, "required": ["questions"]}

MCQ_PROMPT = """You are a CBSE examiner{subject}. Which option is correct?

Question {number}: {text}
{options}

Answer with the option letter (A, B, C or D); options printed as (a)-(d) or (i)-(iv) are A-D in order."""
MCQ_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string", "enum": ["A", "B", "C", "D"]}},
              "required": ["answer"]}

POINTS_PROMPT = """You are a CBSE examiner{subject} writing the marking scheme for one question.

Question {number} ({marks:g} marks): {text}

List the points a full-mark answer makes, one fact or idea per point, in a short sentence each, as an
examiner's marking scheme does. Give {count} points: a student needs any {needed} of them for full marks,
so add other correct points students commonly make. If the question has an internal choice (OR), write
the points for the first question only. Only write points that are correct."""
POINTS_SCHEMA = {"type": "object", "properties": {"points": {"type": "array", "items": {"type": "string"}}},
                 "required": ["points"]}


def chat(model: str, prompt: str, schema: Optional[dict], image: Optional[np.ndarray] = None, num_predict: int = 1500) -> dict:
    message = {"role": "user", "content": prompt}
    if image is not None:
        h, w = image.shape[:2]
        if max(h, w) > 1600:                     # enough for printed text, and quicker
            s = 1600 / max(h, w)
            image = cv2.resize(image, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        message["images"] = [base64.b64encode(cv2.imencode(".png", image)[1]).decode()]
    body = {"model": model, "stream": False, "messages": [message],
            "options": {"temperature": 0, "num_ctx": 8192, "num_predict": num_predict}}
    if schema:
        body["format"] = schema
    if model.startswith("qwen3"):
        body["think"] = False
    req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    for attempt in (1, 2):                       # the model can stall or restart once
        try:
            with urllib.request.urlopen(req, timeout=900) as resp:
                content = json.load(resp)["message"]["content"]
            break
        except (TimeoutError, urllib.error.URLError):
            if attempt == 2:
                raise
            time.sleep(5)
    if not schema:
        return {"text": content}
    try:
        return json.loads(content)
    except json.JSONDecodeError:                 # cut off at num_predict
        return {}


TEXT_PROMPT = ("Copy all the text on this page exactly in order from top to bottom, as a scanner would. The page "
               "often starts in the middle of an answer from the page before: copy those first lines first, "
               "before the next numbered row. Keep the question numbers at the start of their rows.")
EXTRACT_PROMPT = """This is one entry of an official exam marking scheme: the question, then its answer or value points.

\"\"\"
{entry}
\"\"\"

Copy the answer's value points, one per item, exactly as written; leave out the question itself, page
numbers, marks and "Any other relevant point"."""
# A row of the scheme: the question number at the start of a line ("21 Why was...", "34.1 How...").
ROW = re.compile(r"^\s*(\d{1,2}(?:\.\d)?)[.)]?\s+(?=\S)")
ROMAN_ITEM = re.compile(r"(?m)^\s*\((?:i|ii|iii|iv|v|vi|vii|viii|ix|x)\)\s*")
NEEDED = re.compile(r"(?i)\bany\s+(one|two|three|four|five|six)\b")
WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
MARKS = re.compile(r"\b\d+\s*[xX×]\s*\d+(?:\.\d)?\s*=\s*(\d+(?:\.\d)?)")      # "3X1=3"
PLUS_MARKS = re.compile(r"(?m)^\s*\d(?:\s*\+\s*\d)+\s*=\s*(\d+)\s*$")           # "1+1=2" on its own line
ONE_MARK = re.compile(r"\((\d)\)(?=\s)\s*")              # "(1)" after a sub-part's question: its marks
SECTION = re.compile(r"(?im)^\s*section\s*[-–:]?\s*[a-f]\b.*")                   # the next section's heading
MARKS_TAIL = re.compile(r"\s*(?:\b\d+(?:-\d+)?\s?[A-Z]\b\s*)?\d+\s*[xX×]\s*\d+(?:\.\d)?\s*=\s*\d+.*$")


def is_marking_scheme(files: list[Path]) -> bool:
    """True when the upload is an official marking scheme rather than a question paper."""
    pages = [p for _, ps in load_pages(files) for p in ps][:4]
    for img in pages:
        out = chat(MODEL, "Is this page part of an exam MARKING SCHEME (answers or value points for the examiner), "
                          "rather than the question paper students get?", {"type": "object", "properties": {
                              "marking_scheme": {"type": "boolean"}}, "required": ["marking_scheme"]}, img, num_predict=20)
        if out.get("marking_scheme"):
            return True
    return False


def split_entries(text: str) -> dict[str, str]:
    """The scheme's text cut into entries by question number. A number counts as a new row only when
    it follows the order (the next question, or a part of the current one), so numbers inside the
    answers ("1815", "2 marks") don't cut an entry."""
    entries: dict[str, list[str]] = {}
    current, last_main = None, 0
    for line in text.splitlines():
        m = ROW.match(line)
        if m:
            n = m.group(1)
            main = int(n.split(".")[0])
            if current is None or ("." not in n and last_main < main <= last_main + 2) or (
                    "." in n and main in (last_main, last_main + 1)):
                current, last_main = n, main
                entries.setdefault(n, [])
                line = line[m.end():]
        if current:
            entries[current].append(line)
    return {n: "\n".join(lines).strip() for n, lines in entries.items()}


def entry_points(entry: str, part: bool = False) -> list[str]:
    """The value points of one entry: its (i), (ii)... items, both alternatives of an OR."""
    items = ROMAN_ITEM.split(entry)[1:]
    if items:
        out = []
        for item in items:
            # An item ends where "Any two points...", "OR" or the next alternative "(b)" starts.
            item = re.split(r"(?im)^\s*(?:any\s+\w+\s+points?|or\s*$|\(b\)|\d+(?:-\d+)?\s*[A-Z]\b)", item)[0]
            item = re.sub(r"\s+", " ", item).strip()
            item = MARKS_TAIL.sub("", item).strip()      # "... cultural links. 54 H 2X1=2": the page and marks columns
            if item and not re.match(r"(?i)any other relevant", item):
                out.append(item)
        return out
    # A short sub-part answer, "Why was Hickey persecuted? (1) He published gossip...": the answer is
    # what follows the marks.
    m = ONE_MARK.search(entry) if part else None
    if m:
        answer = re.sub(r"\s+", " ", entry[m.end():]).strip()
        if answer and len(answer) < 400:
            return [answer]
    out = chat(JUDGE_MODEL, EXTRACT_PROMPT.format(entry=entry), POINTS_SCHEMA)   # an answer without (i), (ii)
    return [re.sub(r"\s+", " ", str(p)).strip() for p in out.get("points", []) if str(p).strip()
            and not re.match(r"(?i)any other relevant", str(p))]


def read_marking_scheme(files: list[Path], mcq: Optional[list[int]], progress: Callable[[str], None] = print,
                        save_text: Optional[Path] = None) -> dict:
    """The rubric copied from an official marking scheme: {"key": {...}, "points": {...}}, as make_rubric.
    Each page is copied as text first (a page of dense table is too much to structure in one go), then the
    text is cut into questions and the points are taken as printed."""
    pages = [p for _, ps in load_pages(files) for p in ps]
    texts = []
    for i, img in enumerate(pages, 1):
        progress(f"Reading the marking scheme: page {i} of {len(pages)}")
        texts.append(chat(MODEL, TEXT_PROMPT, None, img, num_predict=3000).get("text", ""))
    if save_text:
        save_text.write_text("\n".join(texts), encoding="utf-8")
    return scheme_from_text("\n".join(texts), mcq, progress)


def scheme_from_text(text: str, mcq: Optional[list[int]], progress: Callable[[str], None] = print) -> dict:
    """read_marking_scheme's second step, on the scheme's text. Without mcq, the questions whose answer is
    just an option, "(C) Buddhism", are taken as the MCQs. Also returns each question's marks, from the
    scheme's marks column ("3X1=3"; 1 for an MCQ)."""
    text = "\n".join(line for line in text.splitlines() if "collegedunia" not in line.lower())   # a site's footer
    # The general instructions before the answers are numbered too: start at "Section A" when there is one.
    start = re.search(r"(?im)^.*\bsection\s*[-–:]?\s*a\b", text)
    entries = split_entries(text[start.start():] if start else text)
    key, points, marks = {}, {}, {}
    for i, (n, entry) in enumerate(entries.items(), 1):
        progress(f"Taking the points: question {n} ({i} of {len(entries)})")
        entry = SECTION.split(entry)[0].strip()
        if f"{n}.1" in entries:
            continue                    # a case-based question's source passage: its parts have the points
        if re.search(r"(?i)\boutline\s+map\b|\bmap\s+skill", entry[:300]):
            continue                    # a map question: marked by the teacher on the map
        option = re.match(r"\s*\(([A-D])\)", entry)
        if "." not in n and (int(n) in mcq if mcq is not None else bool(option)):
            m = option or re.search(r"\(([A-D])\)", entry)
            if m:
                key[n] = m.group(1)
            marks[n] = 1
            continue
        m = MARKS.search(entry) or PLUS_MARKS.search(entry) or ("." in n and ONE_MARK.search(entry))
        if m:
            marks[n] = float(m.group(1))
        made = entry_points(entry, "." in n)
        if made:
            m = NEEDED.search(entry)
            points[n] = {"points": made[:20], "needed": WORDS[m.group(1).lower()] if m else None}
    return {"key": key, "points": points, "marks": marks}


def part_marks(scheme: dict[str, float], made: dict) -> dict[str, float]:
    """The marks of each case-based sub-part, added to the test's marks scheme ({"34": 4} gets "34.1": 1...).
    A part whose marks the scheme doesn't print gets its "Any two points" count, and else an equal share of
    what is left of the question's total."""
    out = dict(scheme)
    parts: dict[str, list[str]] = {}
    for n in {*made["marks"], *made["points"]}:
        if "." in n:
            parts.setdefault(n.split(".")[0], []).append(n)
    for main, ns in parts.items():
        got = {n: made["marks"][n] for n in ns if n in made["marks"]}
        for n in ns:
            if n not in got and (made["points"].get(n) or {}).get("needed"):
                got[n] = float(made["points"][n]["needed"])
        rest = [n for n in ns if n not in got]
        if rest and main in scheme:
            share = (scheme[main] - sum(got.values())) / len(rest)
            if share > 0:
                got.update({n: round(share * 2) / 2 for n in rest})
        out.update({n: v for n, v in got.items() if n not in scheme})
    return out


def marks_scheme(marks: dict[str, float]) -> str:
    """{"1": 1, "2": 1, "21": 2, "34.1": 1} -> "1-2:1, 21:2, 34.1:1" (the marks scheme's format). A question
    marked by its parts leaves out its own total."""
    parts = {n.split(".")[0] for n in marks if "." in n}
    items = sorted(((n, v) for n, v in marks.items() if n not in parts), key=lambda x: float(x[0]))
    out, i = [], 0
    while i < len(items):
        j = i
        while (j + 1 < len(items) and "." not in items[j + 1][0] and "." not in items[i][0]
               and int(items[j + 1][0]) == int(items[j][0]) + 1 and items[j + 1][1] == items[i][1]):
            j += 1
        r = items[i][0] if i == j else f"{items[i][0]}-{items[j][0]}"
        out.append(f"{r}:{items[i][1]:g}")
        i = j + 1
    return ", ".join(out)


def read_questions(files: list[Path], progress: Callable[[str], None] = print) -> dict[str, dict]:
    """{"21": {"text": ..., "options": [...]}} for every question printed in the paper."""
    pages = [p for _, ps in load_pages(files) for p in ps]
    found: dict[str, dict] = {}
    for i, img in enumerate(pages, 1):
        progress(f"Reading the questions: page {i} of {len(pages)}")
        for q in chat(MODEL, READ_PROMPT, READ_SCHEMA, img, num_predict=3000).get("questions", []):
            n, text = str(q.get("number", "")), str(q.get("text", "")).strip()
            if not n.isdigit() or not text:
                continue
            letters = re.findall(r"[^\W\d_]", text)
            if sum("ऀ" <= ch <= "ॿ" for ch in letters) > len(letters) * 0.3:
                continue                         # the Hindi version of a bilingual paper: the English one is used
            if n in found and q.get("options") and found[n]["options"]:
                continue                         # the same MCQ read again

            q_found = found.setdefault(n, {"text": "", "options": []})
            q_found["text"] = f"{q_found['text']}\n{text}".strip()   # a question carried over a page
            q_found["options"] += [str(o).strip() for o in q.get("options", []) if str(o).strip()]
    return found


def make_rubric(questions: dict[str, dict], scheme: dict[str, float], mcq: list[int], subject: str = "",
                progress: Callable[[str], None] = print) -> dict:
    """{"key": {"1": "B", ...}, "points": {"21": {"points": [...], "needed": 2}, ...}}"""
    subject = f" for {subject}" if subject else ""
    key, points = {}, {}
    todo = [n for n in sorted(questions, key=int) if n in scheme]
    for i, n in enumerate(todo, 1):
        progress(f"Writing the rubric: question {n} ({i} of {len(todo)})")
        q, marks = questions[n], scheme[n]
        if int(n) in mcq:
            # Options printed in a table or a match-the-columns list are copied into the question text.
            options = "\n".join(q["options"]) or "(the options are in the question above)"
            out = chat(JUDGE_MODEL, MCQ_PROMPT.format(subject=subject, number=n, text=q["text"], options=options),
                       MCQ_SCHEMA, num_predict=50)
            if out.get("answer") in ("A", "B", "C", "D"):
                key[n] = out["answer"]
            continue
        needed = max(1, round(marks))
        out = chat(JUDGE_MODEL, POINTS_PROMPT.format(subject=subject, number=n, marks=marks, text=q["text"],
                                                     count=needed + (1 if needed <= 2 else 2), needed=needed),
                   POINTS_SCHEMA)
        made = [re.sub(r"\s+", " ", str(p)).strip() for p in out.get("points", []) if str(p).strip()][:10]
        if made:
            points[n] = {"points": made, "needed": min(needed, len(made))}
    return {"key": key, "points": points}


if __name__ == "__main__":
    from app import parse_scheme
    sys.stdout.reconfigure(encoding="utf-8")
    qs = read_questions([Path(sys.argv[1])])
    print(json.dumps(qs, indent=1, ensure_ascii=False))
    a, b = sys.argv[3].split("-")
    print(json.dumps(make_rubric(qs, parse_scheme(sys.argv[2]), list(range(int(a), int(b) + 1)), "Social Science"),
                     indent=1, ensure_ascii=False))

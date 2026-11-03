"""Read the marks scheme from a question paper's general instructions.

CBSE-style papers list their sections on the first pages: "Section A - question number 1 to 20
are multiple choice type questions. Each question carries 1 mark." read_scheme() finds that list
with the model, one page at a time, and checks it: the marks must add up to the paper's maximum
marks and the questions must run 1, 2, 3... without gaps. When the check fails the teacher is told,
and corrects the scheme by hand; nothing is guessed.

    python eval/qpaper.py QUESTION_PAPER.pdf
"""
import base64
import json
import sys
import urllib.request
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from booklet import load_pages
from model import MODEL, OLLAMA_URL

PAGES_TO_SEARCH = 4
PROMPT = (
    "This is a page of an exam question paper. If the page has instructions that say which question "
    "numbers are in each section and how many marks each question carries, list every section:\n"
    "- name: the section letter\n- first, last: its first and last question number\n"
    "- marks: marks for EACH question in the section\n"
    "- type: mcq (multiple choice), short, long, case (case/source based) or map\n"
    "Also give max_marks, the paper's maximum marks if printed (0 if not).\n"
    "If the page has no such instructions, return an empty sections list.")
SCHEMA = {"type": "object", "properties": {
    "max_marks": {"type": "number"},
    "sections": {"type": "array", "items": {"type": "object", "properties": {
        "name": {"type": "string"}, "first": {"type": "integer"}, "last": {"type": "integer"},
        "marks": {"type": "number"}, "type": {"type": "string", "enum": ["mcq", "short", "long", "case", "map"]}},
        "required": ["name", "first", "last", "marks", "type"]}}},
    "required": ["max_marks", "sections"]}


def ask_page(img: np.ndarray) -> dict:
    h, w = img.shape[:2]
    if max(h, w) > 1600:                         # enough for printed text, and quicker
        s = 1600 / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, png = cv2.imencode(".png", img)
    body = {"model": MODEL, "stream": False, "format": SCHEMA,
            "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 800},
            "messages": [{"role": "user", "content": PROMPT, "images": [base64.b64encode(png).decode()]}]}
    req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        try:
            return json.loads(json.load(resp)["message"]["content"])
        except (json.JSONDecodeError, KeyError):
            return {"max_marks": 0, "sections": []}


def check(sections: list[dict], max_marks: float) -> list[str]:
    """Problems with the scheme the teacher should know about; empty when it adds up."""
    problems = []
    expected = 1
    for s in sorted(sections, key=lambda s: s["first"]):
        if s["first"] != expected:
            problems.append(f"questions {expected} to {s['first'] - 1} are in no section"
                            if s["first"] > expected else f"section {s['name']} overlaps the one before")
        if s["last"] < s["first"]:
            problems.append(f"section {s['name']} ends before it starts")
        expected = s["last"] + 1
    total = sum((s["last"] - s["first"] + 1) * s["marks"] for s in sections)
    if max_marks and abs(total - max_marks) > 1e-6:
        problems.append(f"the sections add up to {total:g} marks, but the paper says {max_marks:g}")
    if not max_marks:
        problems.append("the paper's maximum marks were not found, so the total could not be checked")
    return problems


def scheme_text(sections: list[dict]) -> str:
    """In the results page's format: "1-20:1, 21-24:2"."""
    parts = []
    for s in sorted(sections, key=lambda s: s["first"]):
        r = str(s["first"]) if s["first"] == s["last"] else f"{s['first']}-{s['last']}"
        parts.append(f"{r}:{s['marks']:g}")
    return ", ".join(parts)


def read_scheme(files: list[Path], progress: Callable[[str], None] = print) -> dict:
    pages = [p for _, ps in load_pages(files) for p in ps][:PAGES_TO_SEARCH]
    max_marks = 0.0
    for i, img in enumerate(pages, 1):
        progress(f"Looking for the instructions on page {i} of {len(pages)}")
        found = ask_page(img)
        max_marks = max_marks or float(found.get("max_marks") or 0)
        sections = [s for s in found.get("sections", []) if s.get("first") and s.get("last")]
        if sections:
            # Some papers print the maximum marks on the cover page, before the instructions.
            return {"found_on_page": i, "max_marks": max_marks, "sections": sections,
                    "scheme": scheme_text(sections), "problems": check(sections, max_marks),
                    "mcq": [n for s in sections if s["type"] == "mcq" for n in range(s["first"], s["last"] + 1)]}
    return {"found_on_page": None, "max_marks": max_marks, "sections": [], "scheme": "", "mcq": [],
            "problems": [f"no list of sections and marks was found on the first {len(pages)} pages; "
                         "type the marks scheme by hand"]}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(read_scheme([Path(p) for p in sys.argv[1:]]), indent=1, ensure_ascii=False))

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
from pathlib import Path
from typing import Callable, Optional

from model import OLLAMA_URL, read_answer
from points import evidence_found

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
SCHEMA = {"type": "object", "properties": {"made": {"type": "boolean"}, "quote": {"type": "string"}},
          "required": ["made", "quote"]}


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


def suggest(made: int, needed: int, marks: float) -> float:
    return round(min(made, needed) / needed * marks * 2) / 2


def grade(result: dict, crops_dir: Path, points: dict[str, dict], scheme: dict[str, float],
          only: Optional[str] = None, progress: Callable[[str], None] = print) -> dict:
    """Add a "grading" entry to each question with marking points (in place) and return result.

    points: {"25": {"points": ["...", ...], "needed": 3}}; only: grade just this question number."""
    todo = [q for q in result["questions"]
            if q["number"] in points and points[q["number"]].get("points") and (only is None or q["number"] == only)]
    for i, q in enumerate(todo, 1):                     # 1. read every answer (the reading model)
        progress(f"Reading long answers: {i} of {len(todo)}")
        texts = []
        for c in q["crops"]:
            text = read_answer(str(crops_dir / c["file"]), q["number"])["text"] or ""
            if text.strip():
                texts.append(text.strip())
        q["transcript"] = "\n".join(texts)
    total = sum(len(points[q["number"]]["points"]) for q in todo)
    done = 0
    for q in todo:                                      # 2. judge every point (the judge model)
        spec = points[q["number"]]
        marks = scheme.get(q["number"]) or scheme.get(q["number"].split(".")[0]) or 0
        needed = max(1, min(int(spec.get("needed") or len(spec["points"])), len(spec["points"])))
        checked = []
        for p in spec["points"]:
            done += 1
            progress(f"Checking marking points: {done} of {total}")
            if not q["transcript"]:
                checked.append({"point": p, "made": False, "quote": "", "note": ""})
            else:
                checked.append(judge(q["transcript"], p, q["number"], marks))
        made = sum(c["made"] for c in checked)
        q["grading"] = {"points": checked, "made": made, "needed": needed, "out_of": marks,
                        "suggested": suggest(made, needed, marks) if marks else None,
                        "empty": not q["transcript"]}
    return result

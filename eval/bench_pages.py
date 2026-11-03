"""Benchmark whole-page reading: does the model put each answer under the right question number?

Reads every upright page image in a folder with one model, saves the raw output, and scores the
question labels it found against a truth file (see C:\\ocr_eval\\try\\bench\\truth_labels.json):

    {"page_03": {"labels": ["13", "10", ...], "hidden": ["10", ...]}, ...}

Labels are compared by main question number (34.1 -> 34), as a multiset per page:
    right      a label the student wrote and the model found
    invented   a number the model gave that the student did not write there: the dangerous case,
               an answer silently filed under the wrong question
    missed     a visible label the model did not give (hidden ones are not counted)
    flagged    the model answered "?" for a label it could not read, or its output broke: safe,
               goes to the teacher

Usage:  python eval/bench_pages.py --pages DIR --truth truth_labels.json --model qwen2.5vl:7b --out-dir DIR
        add --rescore to score saved outputs again without calling the model.
"""
import argparse
import base64
import csv
import json
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

from model import OLLAMA_URL

PROMPT = (
    "This is one page of a student's handwritten exam answer booklet. Students write the question number "
    "in the left margin (like 1., 13., 34.1, 25. b)) and then the answer.\n"
    "Transcribe every answer on the page, exactly as written:\n"
    "- question: the question number written in the margin, exactly as written. If the number is hidden, "
    "covered or unreadable, use \"?\". Never guess a number.\n"
    "- If the page starts by continuing an answer from the previous page (no question number), use "
    "\"continued\" for that part.\n"
    "- Numbered points inside an answer (1., 2., circled 1, (i)) belong to that answer; they are not new questions.\n"
    "- Leave out anything crossed out. Ignore red-ink teacher marks, ticks and page numbers.\n"
    "Return the answers in the order they appear.")
SCHEMA = {"type": "object", "properties": {"answers": {"type": "array", "items": {
    "type": "object", "properties": {"question": {"type": "string"}, "answer": {"type": "string"}},
    "required": ["question", "answer"]}}}, "required": ["answers"]}
BROKEN = "<broken output>"
FIELDS = ["page", "seconds", "truth", "predicted", "right", "invented", "missed", "flagged", "hidden"]


def main_number(label: str) -> str | None:
    m = re.match(r"\s*(?:Q\.?|Ans\.?)?\s*(\d+)", label, re.I)
    return str(int(m.group(1))) if m else None


def read_page(path: Path, model: str) -> tuple[list[dict], float]:
    body = {"model": model, "stream": False, "format": SCHEMA,
            "messages": [{"role": "user", "content": PROMPT,
                          "images": [base64.b64encode(path.read_bytes()).decode()]}],
            "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 3000}}
    if model.startswith("qwen3"):
        body["think"] = False
    start = time.time()
    req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as resp:
        content = json.load(resp)["message"]["content"]
    try:
        answers = json.loads(content)["answers"]
    except (json.JSONDecodeError, KeyError):
        # Runaway output cut off by the context limit: the page failed, which the teacher would see.
        answers = [{"question": BROKEN, "answer": content[:300]}]
    return answers, time.time() - start


def score(truth: dict, answers: list[dict]) -> dict:
    labels = [a["question"].strip() for a in answers]
    want = Counter(main_number(t) for t in truth["labels"])
    hidden = Counter(main_number(t) for t in truth["hidden"])
    got = Counter(n for n in map(main_number, labels) if n)
    right = got & want
    missed = (want - right) - hidden
    return {"right": sum(right.values()), "invented": sum((got - want).values()),
            "missed": sum(missed.values()), "flagged": sum(l in ("?", BROKEN) for l in labels),
            "hidden": sum(hidden.values()), "predicted": " | ".join(labels),
            "truth": " | ".join(truth["labels"]), "invented_list": sorted((got - want).elements())}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=Path, required=True)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--rescore", action="store_true")
    args = ap.parse_args()
    truth = {k: v for k, v in json.loads(args.truth.read_text(encoding="utf-8")).items() if not k.startswith("_")}
    raw_dir = args.out_dir / ("raw_" + args.model.replace(":", "_"))
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for page in sorted(truth):
        raw = raw_dir / f"{page}.json"
        if args.rescore or raw.exists():
            saved = json.loads(raw.read_text(encoding="utf-8"))
        else:
            answers, secs = read_page(args.pages / f"{page}.png", args.model)
            saved = {"seconds": round(secs, 1), "answers": answers}
            raw.write_text(json.dumps(saved, ensure_ascii=False, indent=1), encoding="utf-8")
        s = score(truth[page], saved["answers"])
        rows.append({"page": page, "seconds": saved["seconds"], **{k: s[k] for k in FIELDS[2:]}})
        mark = "  <-- INVENTED " + ",".join(s["invented_list"]) if s["invented"] else ""
        print(f"{page} {saved['seconds']:5.0f}s  truth [{s['truth']}]  model [{s['predicted']}]{mark}", flush=True)

    out = args.out_dir / f"labels_{args.model.replace(':', '_')}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    total = lambda k: sum(r[k] for r in rows)
    labels = sum(len(truth[p]["labels"]) for p in truth)
    print(f"\n{args.model}: {len(rows)} pages, {total('seconds') / len(rows):.0f} s/page")
    print(f"  labels right: {total('right')} of {labels} ({total('hidden')} hidden under patches); "
          f"visible ones missed: {total('missed')}; flagged '?': {total('flagged')}")
    print(f"  INVENTED (wrong number, silent): {total('invented')} on "
          f"{sum(r['invented'] > 0 for r in rows)} pages")
    print(f"  saved {out}")


if __name__ == "__main__":
    main()

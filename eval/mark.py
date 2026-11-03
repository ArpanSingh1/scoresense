"""Mark predictions.csv against the teacher's answer key; total marks per sheet.

The key is a JSON file, one entry per question id (the ids in crops.csv):
    {"Q1": {"type": "mcq", "options": "letters", "answer": "A", "marks": 1},
     "Q2": {"type": "mcq", "options": "roman", "answer": "iii", "marks": 1},
     "Q3": {"type": "text", "answer": ["Meghalaya", "Megalaya"], "marks": 1},
     "Q4": {"type": "manual", "marks": 5}}
  mcq     the option label the answer starts with is compared to "answer"
          ("letters" = A-D, "roman" = i-iv); model.check_option's rules apply
  text    compared ignoring case, spaces and punctuation; "answer" is one
          accepted answer or a list of them
  manual  always sent to the teacher (long answers)

Each answer gets a result:
  correct  read confidently and matches the key: full marks
  wrong    read confidently and does not match: 0 marks
  review   the teacher must look: low confidence, no text, no option label,
           several options written (often crossed out), or a manual question.
           Marks stay pending, so a sheet's total is a minimum until reviewed.

"wrong" is decided by the model alone, so a misread can cost a student marks:
check the wrong list before releasing marks until the review threshold has been
calibrated on real labelled crops.

With --sheet-key (crop_id, sheet_file, question_id, as cut_crops.py writes it),
marks are totalled per sheet; without it, per crop only.

Usage:  python eval/mark.py --key KEY.json [--predictions PATH] [--crops-list PATH]
                            [--sheet-key PATH] [--out marks.csv]
"""
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

from model import LETTER_OPTIONS, REVIEW_THRESHOLD, ROMAN_OPTIONS, check_option

HERE = Path(__file__).parent
OPTION_SETS = {"letters": LETTER_OPTIONS, "roman": ROMAN_OPTIONS}


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_key(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        key = json.load(f)
    for qid, q in key.items():
        kind = q.get("type")
        if kind not in ("mcq", "text", "manual"):
            sys.exit(f"{path}: {qid} needs type mcq, text or manual, got {kind!r}")
        if not isinstance(q.get("marks"), (int, float)) or q["marks"] < 0:
            sys.exit(f"{path}: {qid} needs a number of marks")
        if kind == "mcq":
            options = OPTION_SETS.get(q.get("options"))
            if options is None:
                sys.exit(f"{path}: {qid} needs options \"letters\" or \"roman\"")
            if str(q.get("answer", "")).lower() not in {o.lower() for o in options}:
                sys.exit(f"{path}: {qid} answer {q.get('answer')!r} is not one of {options}")
        if kind == "text" and not q.get("answer"):
            sys.exit(f"{path}: {qid} needs an answer (or a list of accepted answers)")
    return key


def simplify(s: str) -> str:
    return re.sub(r"[\W_]+", "", s.casefold())


def parse_conf(value: Optional[str]) -> Optional[float]:
    return float(value) if value else None


def mark(q: dict, text: str, confidence: Optional[float]) -> tuple[str, str, list[str]]:
    """Return (result, what was read as the answer, reasons for review)."""
    if q["type"] == "manual":
        return "review", text, ["long answer: marked by the teacher"]
    reasons = []
    if not text:
        reasons.append("no text")
    if confidence is None or confidence < REVIEW_THRESHOLD:
        reasons.append("low confidence")
    if q["type"] == "mcq":
        _, option, option_reasons = check_option(text, OPTION_SETS[q["options"]])
        reasons += option_reasons
        read = option or ""
        matches = option is not None and option.lower() == str(q["answer"]).lower()
    else:
        accepted = q["answer"] if isinstance(q["answer"], list) else [q["answer"]]
        read = text
        matches = simplify(text) in {simplify(a) for a in accepted}
    if reasons:
        return "review", read, reasons
    return ("correct" if matches else "wrong"), read, []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", type=Path, required=True, help="answer key JSON")
    ap.add_argument("--predictions", type=Path, default=HERE / "predictions.csv")
    ap.add_argument("--crops-list", type=Path, default=HERE / "dataset" / "crops.csv")
    ap.add_argument("--sheet-key", type=Path, help="crop_id -> sheet_file (from cut_crops.py --key)")
    ap.add_argument("--out", type=Path, default=HERE / "marks.csv")
    args = ap.parse_args()

    key = load_key(args.key)
    questions = {r["crop_id"]: r["question_id"] for r in read_csv(args.crops_list)}
    preds = {r["crop_id"]: r for r in read_csv(args.predictions)}
    sheets = ({r["crop_id"]: r["sheet_file"] for r in read_csv(args.sheet_key)}
              if args.sheet_key else {})

    unknown = sorted({q for q in questions.values() if q not in key})
    if unknown:
        sys.exit(f"{args.key} has no entry for {', '.join(unknown)}; nothing was marked.")
    missing = sorted(c for c in questions if c not in preds)
    if missing:
        print(f"NOTE: no prediction for {len(missing)} crops; they go to review: {', '.join(missing)}\n")

    rows = []
    for crop_id, qid in questions.items():
        q, p = key[qid], preds.get(crop_id, {})
        text = p.get("predicted_text", "")
        result, read, reasons = mark(q, text, parse_conf(p.get("confidence_min")))
        rows.append({
            "sheet_file": sheets.get(crop_id, ""), "question_id": qid, "crop_id": crop_id,
            "model_read": text, "answer_read": read,
            "key_answer": "" if q["type"] == "manual" else (
                " | ".join(q["answer"]) if isinstance(q["answer"], list) else q["answer"]),
            "result": result,
            "marks": q["marks"] if result == "correct" else (0 if result == "wrong" else ""),
            "out_of": q["marks"], "review_reasons": "; ".join(reasons),
        })
    # Q2 before Q10: compare the numbers in a question id as numbers.
    natural = lambda s: [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", s)]
    rows.sort(key=lambda r: (r["sheet_file"], natural(r["question_id"]), r["crop_id"]))

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    counts = defaultdict(int)
    for r in rows:
        counts[r["result"]] += 1
    print(f"{len(rows)} answers: {counts['correct']} correct, {counts['wrong']} wrong, "
          f"{counts['review']} for the teacher to review ({counts['review'] / len(rows):.0%})")
    print(f"Wrote {args.out}\n")

    if sheets:
        totals = defaultdict(lambda: {"marks": 0, "out_of": 0, "pending": 0, "review": 0})
        for r in rows:
            t = totals[r["sheet_file"]]
            t["out_of"] += r["out_of"]
            if r["result"] == "review":
                t["pending"] += r["out_of"]
                t["review"] += 1
            else:
                t["marks"] += r["marks"]
        print(f"{'sheet':<30}{'marks':>7}{'out of':>8}{'pending':>9}{'to review':>11}")
        for sheet, t in sorted(totals.items()):
            print(f"{sheet:<30}{t['marks']:>7g}{t['out_of']:>8g}{t['pending']:>9g}{t['review']:>11}")
        print()

    for title, result in (("Wrong (decided by the model: check before releasing marks)", "wrong"),
                          ("For review", "review")):
        listed = [r for r in rows if r["result"] == result]
        if listed:
            print(title)
            for r in listed:
                why = f"  [{r['review_reasons']}]" if r["review_reasons"] else ""
                print(f"  {r['sheet_file'] or r['crop_id']:<22}{r['question_id']:<6}"
                      f"read {r['answer_read']!r:<14} key {r['key_answer']!r}{why}")
            print()


if __name__ == "__main__":
    main()

"""Score predictions.csv against labels.csv: CER table and review-threshold sweep.

This is the only script that reads ground_truth_text. Question ids come from
the answer-free crops.csv. Refuses to score unless labels.csv has exactly one
row per crop in crops.csv.

CER = Levenshtein(prediction, ground_truth) / len(ground_truth). Raw CER
collapses and strips whitespace; case and punctuation count. CER_nows removes
all whitespace from both sides first, so "x = 7" vs "x=7" scores 0 (but so does
"photo synthesis" vs "photosynthesis"). An empty prediction scores CER 1.0.
Per-question CER is micro-averaged (total edits / total GT chars).

Label values (as label.html writes them):
  ""             not labelled yet: left out of every number, listed in a NOTE
  "[blank]"      the student wrote nothing: ground truth is "", so only an
                 empty prediction is exact
  "[illegible]"  the labeller couldn't read it: no ground truth, left out of
                 every number, listed in a NOTE

Sweep: a crop is sent to review when its prediction is empty, its confidence
is missing, or confidence < threshold. "Wrong" means not an exact match.
The sweep uses confidence_min (the review signal) unless --signal mean.

Usage:  python eval/score.py [--crops-list PATH] [--labels PATH] [--predictions PATH]
                             [--per-crop] [--signal min|mean]
"""
import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

HERE = Path(__file__).parent
THRESHOLDS = [round(0.50 + 0.05 * i, 2) for i in range(10)]  # 0.50 .. 0.95
BLANK, ILLEGIBLE = "[blank]", "[illegible]"


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalize(s: str) -> str:
    return " ".join(s.split())


def strip_ws(s: str) -> str:
    return "".join(s.split())


def rate(edits: int, chars: int) -> float:
    # No ground-truth chars (a [blank] answer): any predicted text is fully wrong.
    return edits / chars if chars else float(edits > 0)


def parse_conf(value: Optional[str]) -> Optional[float]:
    return float(value) if value else None


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops-list", type=Path, default=HERE / "dataset" / "crops.csv")
    ap.add_argument("--labels", type=Path, default=HERE / "dataset" / "labels.csv")
    ap.add_argument("--predictions", type=Path, default=HERE / "predictions.csv")
    ap.add_argument("--per-crop", action="store_true", help="also print CER for each crop")
    ap.add_argument("--signal", choices=["min", "mean"], default="min",
                    help="confidence column the sweep uses (default: min)")
    args = ap.parse_args()

    questions = {r["crop_id"]: r["question_id"] for r in read_csv(args.crops_list)}
    labels = read_csv(args.labels)
    preds = {r["crop_id"]: r for r in read_csv(args.predictions)}

    label_ids = [r["crop_id"] for r in labels]
    if len(label_ids) != len(set(label_ids)) or set(label_ids) != set(questions):
        dupes = sorted(i for i, n in Counter(label_ids).items() if n > 1)
        sys.exit(f"labels.csv ({len(label_ids)} rows) doesn't match crops.csv ({len(questions)} crops): "
                 f"missing labels {sorted(set(questions) - set(label_ids))}, "
                 f"labels for unknown crops {sorted(set(label_ids) - set(questions))}, "
                 f"duplicate labels {dupes}")
    unlabelled = [r["crop_id"] for r in labels if not r["ground_truth_text"].strip()]
    illegible = [r["crop_id"] for r in labels if r["ground_truth_text"].strip() == ILLEGIBLE]
    if unlabelled:
        print(f"NOTE: {len(unlabelled)} of {len(labels)} crops are not labelled yet and are "
              f"left out: {', '.join(unlabelled)}")
    if illegible:
        print(f"NOTE: {len(illegible)} of {len(labels)} crops were illegible to the labeller and "
              f"are left out: {', '.join(illegible)}")
    if unlabelled or illegible:
        print()
    left_out = set(unlabelled) | set(illegible)

    stats = defaultdict(lambda: {"n": 0, "predicted": 0, "exact": 0,
                                 "edits": 0, "chars": 0, "edits_nows": 0, "chars_nows": 0})
    crops = []
    for row in labels:
        if row["crop_id"] in left_out:
            continue
        qid = questions[row["crop_id"]]
        p = preds.get(row["crop_id"], {})
        label = row["ground_truth_text"].strip()
        gt = "" if label == BLANK else normalize(label)
        pred = normalize(p.get("predicted_text", ""))
        edits = levenshtein(pred, gt)
        edits_nows = levenshtein(strip_ws(pred), strip_ws(gt))
        crops.append({
            "crop_id": row["crop_id"], "qid": qid, "gt": gt, "pred": pred,
            "exact": pred == gt,
            "cer": rate(edits, len(gt)), "cer_nows": rate(edits_nows, len(strip_ws(gt))),
            "min": parse_conf(p.get("confidence_min")),
            "mean": parse_conf(p.get("confidence_mean")),
        })
        for key in (qid, "ALL"):
            s = stats[key]
            s["n"] += 1
            s["predicted"] += bool(pred)
            s["exact"] += pred == gt
            s["edits"] += edits
            s["chars"] += len(gt)
            s["edits_nows"] += edits_nows
            s["chars_nows"] += len(strip_ws(gt))

    header = f"{'question':<10}{'n':>4}{'predicted':>11}{'exact':>7}{'CER':>8}{'CER_nows':>10}"
    print(header)
    print("-" * len(header))
    for key in sorted(k for k in stats if k != "ALL") + ["ALL"]:
        if key == "ALL":
            print("-" * len(header))
        s = stats[key]
        print(f"{key:<10}{s['n']:>4}{s['predicted']:>11}{s['exact']:>7}"
              f"{rate(s['edits'], s['chars']):>8.3f}{rate(s['edits_nows'], s['chars_nows']):>10.3f}")

    if args.per_crop:
        print()
        print(f"{'crop':<10}{'question':<10}{'CER':>7}{'CER_nows':>10}{'conf_min':>10}{'conf_mean':>11}"
              "  ground_truth -> predicted")
        for c in crops:
            cmin = "" if c["min"] is None else f"{c['min']:.4f}"
            cmean = "" if c["mean"] is None else f"{c['mean']:.4f}"
            print(f"{c['crop_id']:<10}{c['qid']:<10}{c['cer']:>7.3f}{c['cer_nows']:>10.3f}"
                  f"{cmin:>10}{cmean:>11}  {c['gt']!r} -> {c['pred']!r}")

    wrong = [c for c in crops if not c["exact"]]
    right = [c for c in crops if c["exact"]]
    print()
    print(f"Review sweep on confidence_{args.signal} "
          f"({len(wrong)} wrong, {len(right)} correct, {len(crops)} crops)")
    sweep_header = f"{'threshold':<11}{'wrong flagged':>15}{'correct flagged':>17}{'sent to review':>16}"
    print(sweep_header)
    print("-" * len(sweep_header))
    for t in THRESHOLDS:
        flagged = lambda c: not c["pred"] or c[args.signal] is None or c[args.signal] < t
        wf = sum(map(flagged, wrong))
        rf = sum(map(flagged, right))
        share = (wf + rf) / len(crops) if crops else 0.0
        print(f"{t:<11.2f}{f'{wf}/{len(wrong)}':>15}{f'{rf}/{len(right)}':>17}{share:>15.0%}")


if __name__ == "__main__":
    main()

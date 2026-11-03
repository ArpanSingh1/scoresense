"""Run read_answer() on every crop in crops.csv and write predictions.csv.

Reads only the answer-free crop list (crop_id, question_id, image_path) and
refuses any file with other columns, so ground truth never reaches this script.
image_path is relative to the crop list's folder unless absolute.
predicted_text / confidence columns are left empty when the model returns None.
--out must be outside the crop list's folder (for real crops, use
C:\\ocr_eval_results), so predictions never sit next to what the labeller gets.

Usage:  python eval/predict.py [--crops-list PATH] [--out PATH]
"""
import argparse
import csv
import os
import sys
from pathlib import Path

from model import read_answer

HERE = Path(__file__).parent
CROP_LIST_COLUMNS = ["crop_id", "question_id", "image_path"]


def fmt(value) -> str:
    return "" if value is None else f"{value:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops-list", type=Path, default=HERE / "dataset" / "crops.csv")
    ap.add_argument("--out", type=Path, default=HERE / "predictions.csv")
    args = ap.parse_args()

    with open(args.crops_list, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != CROP_LIST_COLUMNS:
            sys.exit(f"{args.crops_list}: expected columns {CROP_LIST_COLUMNS}, got "
                     f"{reader.fieldnames}. predict.py only reads the answer-free crop list.")
        crops = list(reader)

    if args.out.resolve().is_relative_to(args.crops_list.resolve().parent):
        sys.exit(f"--out must be outside {args.crops_list.resolve().parent}: predictions are "
                 "never stored next to the crops the labeller receives.")

    onedrive = os.environ.get("OneDrive")
    if onedrive and args.crops_list.resolve().is_relative_to(Path(onedrive).resolve()):
        print(f"WARNING: {args.crops_list.parent} is inside OneDrive and may sync to the cloud. "
              "Keep real student crops in a local folder.", file=sys.stderr)

    rows = []
    for row in crops:
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = args.crops_list.parent / image_path
        result = read_answer(str(image_path), row["question_id"])
        rows.append({
            "crop_id": row["crop_id"],
            "predicted_text": "" if result["text"] is None else result["text"],
            "confidence_min": fmt(result["confidence"]),
            "confidence_mean": fmt(result["confidence_mean"]),
        })

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["crop_id", "predicted_text",
                                               "confidence_min", "confidence_mean"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} predictions to {args.out}")


if __name__ == "__main__":
    main()

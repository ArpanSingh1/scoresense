"""Cut answer regions from scanned answer sheets and write crops.csv.

Each sheet image (PNG/JPG) is one page. Regions come from a JSON file per
assessment, as fractions of the TEMPLATE page width/height; nothing is detected
or guessed:
    {"Q1": {"page": 1, "x0": 0.10, "y0": 0.20, "x1": 0.90, "y1": 0.35}, ...}

Alignment: with --template (an image of the blank printed sheet), each scan is
matched to the template by its printed content (ORB features + RANSAC
homography) and warped onto it before cutting. Sheets that fail alignment are
listed in needs_manual_check.csv while the remaining sheets continue.
Without --template, scans are cut as-is and must line up exactly.

Every sheet in --sheets is treated as page --page; regions for other pages are
skipped. For multi-page assessments, run once per page folder with --append.

Crop ids are shuffled within each batch, so the labeller can't tell which
crops came from the same sheet. Writes:
  <out-dir>/crops/crop_NNN.png
  <out-dir>/crops.csv   (crop_id, question_id, image_path)
  <out-dir>/labels.csv  (crop_id, ground_truth_text) with blank rows to fill in
  --key                 (crop_id, sheet_file, question_id): private, outside
                        <out-dir>; used to delete a writer's crops on withdrawal
No answer text is written anywhere.

Checks each crop and prints warnings for blank crops and ink touching an edge.
Both checks ignore long straight lines (box borders, ruled lines), so a printed
answer box doesn't count as writing.

Usage:  python eval/cut_crops.py --sheets DIR --regions FILE.json
                                 [--template BLANK.png] [--key PATH]
                                 [--out-dir C:\\ocr_eval] [--page 1] [--append]
"""
import argparse
import csv
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

MARGIN_PX = 5
INK_THRESHOLD = 128     # grayscale below this counts as ink
BLANK_MAX_INK_PX = 10   # absolute count: a fraction of a wide box misses answers like "7"
EDGE_BAND_PX = 2
EDGE_MIN_INK_PX = 3     # ignore a few stray dark pixels at an edge
SHEET_EXTS = {".png", ".jpg", ".jpeg"}

ALIGN_WIDTH = 1000      # feature matching runs on images resized to this width
MIN_GOOD_MATCHES = 15
MIN_INLIERS = 12
MAX_ROTATION_DEG = 10
MIN_SCALE, MAX_SCALE = 0.70, 1.30
MAX_SKEW = 0.15


def load_regions(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        regions = json.load(f)
    for qid, r in regions.items():
        missing = {"page", "x0", "y0", "x1", "y1"} - r.keys()
        if missing:
            sys.exit(f"{path}: {qid} is missing {sorted(missing)}")
        if not (0 <= r["x0"] < r["x1"] <= 1 and 0 <= r["y0"] < r["y1"] <= 1):
            sys.exit(f"{path}: {qid} needs 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1, got {r}")
    return regions


def read_image(path: Path) -> np.ndarray:
    # imdecode instead of imread so non-ASCII Windows paths work
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"Could not read image: {path}")
    return img


def align(img: np.ndarray, template: np.ndarray) -> tuple[np.ndarray | None, dict]:
    """Warp scan to template; return image and transform metrics/reason."""
    th, tw = template.shape[:2]
    sh, sw = img.shape[:2]
    s_t, s_s = ALIGN_WIDTH / tw, ALIGN_WIDTH / sw
    t_small = cv2.resize(cv2.cvtColor(template, cv2.COLOR_BGR2GRAY), None, fx=s_t, fy=s_t)
    g_small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None, fx=s_s, fy=s_s)

    orb = cv2.ORB_create(5000)
    kp_s, des_s = orb.detectAndCompute(g_small, None)
    kp_t, des_t = orb.detectAndCompute(t_small, None)
    if des_s is None or des_t is None:
        return None, {"match_count": 0, "rotation_deg": "", "scale": "",
                      "translation_x_px": "", "translation_y_px": "", "reason": "no features found"}
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des_s, des_t, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < MIN_GOOD_MATCHES:
        return None, {"match_count": len(good), "rotation_deg": "", "scale": "",
                      "translation_x_px": "", "translation_y_px": "",
                      "reason": f"only {len(good)} good matches (need {MIN_GOOD_MATCHES})"}

    src = np.float32([kp_s[m.queryIdx].pt for m in good]) / s_s
    dst = np.float32([kp_t[m.trainIdx].pt for m in good]) / s_t
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0
    if H is None or inliers < MIN_INLIERS:
        return None, {"match_count": len(good), "rotation_deg": "", "scale": "",
                      "translation_x_px": "", "translation_y_px": "",
                      "reason": f"only {inliers} RANSAC inliers (need {MIN_INLIERS})"}

    rotation = math.degrees(math.atan2(H[1, 0], H[0, 0]))
    # Normalize out expected resolution difference before reporting scale.
    scale = math.hypot(H[0, 0], H[1, 0]) / (tw / sw)
    centre = cv2.perspectiveTransform(np.float32([[[sw / 2, sh / 2]]]), H)[0, 0]
    offset = centre - np.float32([tw / 2, th / 2])
    # Check the projective map's local axes near the page center for skew.
    pts = np.float32([[[sw / 2, sh / 2], [sw / 2 + 10, sh / 2], [sw / 2, sh / 2 + 10]]])
    mapped = cv2.perspectiveTransform(pts, H)[0]
    vx, vy = mapped[1] - mapped[0], mapped[2] - mapped[0]
    skew = abs(float(np.dot(vx, vy))) / max(1e-9, float(np.linalg.norm(vx) * np.linalg.norm(vy)))
    metrics = {"match_count": len(good), "rotation_deg": round(rotation, 4),
               "scale": round(scale, 5), "translation_x_px": round(float(offset[0]), 2),
               "translation_y_px": round(float(offset[1]), 2), "reason": ""}
    reason = None
    if abs(rotation) > MAX_ROTATION_DEG:
        reason = f"rotation {rotation:.2f} deg exceeds {MAX_ROTATION_DEG} deg"
    elif not MIN_SCALE <= scale <= MAX_SCALE:
        reason = f"scale {scale:.3f} outside {MIN_SCALE:.2f}..{MAX_SCALE:.2f}"
    elif skew > MAX_SKEW:
        reason = f"unstable homography skew {skew:.3f} exceeds {MAX_SKEW:.2f}"
    if reason:
        metrics["reason"] = reason
        return None, metrics
    warped = cv2.warpPerspective(img, H, (tw, th), flags=cv2.INTER_LINEAR,
                                 borderValue=(255, 255, 255))
    return warped, metrics


def cut(img: np.ndarray, r: dict) -> np.ndarray:
    h, w = img.shape[:2]
    x0 = max(0, int(r["x0"] * w) - MARGIN_PX)
    y0 = max(0, int(r["y0"] * h) - MARGIN_PX)
    x1 = min(w, int(round(r["x1"] * w)) + MARGIN_PX)
    y1 = min(h, int(round(r["y1"] * h)) + MARGIN_PX)
    return img[y0:y1, x0:x1]


def handwriting_ink(crop: np.ndarray) -> np.ndarray:
    """Ink mask with long straight lines (box borders, ruled lines) removed."""
    ink = (cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) < INK_THRESHOLD).astype(np.uint8)
    h, w = ink.shape
    horiz = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, max(40, w // 4)), np.uint8))
    vert = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((max(40, h // 2), 1), np.uint8))
    return (ink & ~(horiz | vert)).astype(bool)


def check(crop: np.ndarray) -> list[str]:
    ink = handwriting_ink(crop)
    if ink.sum() < BLANK_MAX_INK_PX:
        return ["blank (almost no ink)"]
    b = EDGE_BAND_PX
    edges = {"left": ink[:, :b], "right": ink[:, -b:], "top": ink[:b, :], "bottom": ink[-b:, :]}
    touching = [name for name, band in edges.items() if band.sum() >= EDGE_MIN_INK_PX]
    return [f"ink touches {', '.join(touching)} edge (possible clipping)"] if touching else []


def append_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    new_file = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", type=Path, required=True, help="folder of scanned sheets")
    ap.add_argument("--regions", type=Path, required=True, help="region JSON for this assessment")
    ap.add_argument("--template", type=Path,
                    help="image of the blank printed sheet; scans are aligned to it before cutting")
    ap.add_argument("--key", type=Path, default=Path(r"C:\ocr_eval_private\sheet_key.csv"),
                    help="crop_id -> sheet_file, question_id; must be outside --out-dir")
    ap.add_argument("--out-dir", type=Path, default=Path(r"C:\ocr_eval"))
    ap.add_argument("--page", type=int, default=1, help="page number of every sheet in --sheets")
    ap.add_argument("--append", action="store_true",
                    help="continue numbering after an existing crops.csv instead of refusing")
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    crops_dir = out_dir / "crops"
    crops_csv = out_dir / "crops.csv"
    labels_csv = out_dir / "labels.csv"
    if args.key.resolve().is_relative_to(out_dir):
        sys.exit(f"--key must be outside {out_dir} so it is never handed out with the crops.")

    regions = {q: r for q, r in load_regions(args.regions).items() if r["page"] == args.page}
    if not regions:
        sys.exit(f"No regions for page {args.page} in {args.regions}")
    sheets = sorted(p for p in args.sheets.iterdir() if p.suffix.lower() in SHEET_EXTS)
    if not sheets:
        sys.exit(f"No PNG/JPG sheets in {args.sheets}")

    existing = []
    if crops_csv.exists():
        if not args.append:
            sys.exit(f"{crops_csv} already exists. Use --append to add to it; nothing was written.")
        with open(crops_csv, newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
    elif any(crops_dir.glob("crop_*.png")):
        sys.exit(f"{crops_dir} already has crop images but no crops.csv; nothing was written.")
    elif labels_csv.exists():
        sys.exit(f"{labels_csv} exists but there is no crops.csv, so its rows can't match these "
                 "crops. Remove it (this script writes a matching one); nothing was written.")
    next_n = 1 + max((int(re.sub(r"\D", "", r["crop_id"])) for r in existing), default=0)

    # Keep a report row for every input sheet; failures don't block valid sheets.
    template = read_image(args.template) if args.template else None
    pages, alignment_rows, manual_rows = [], [], []
    for sheet in sheets:
        img = read_image(sheet)
        if template is None:
            pages.append((sheet, img))
            alignment_rows.append({"sheet_file": sheet.name, "match_count": "",
                                  "rotation_deg": "", "scale": "",
                                  "translation_x_px": "", "translation_y_px": "",
                                  "status": "unaligned", "reason": "no template supplied"})
            continue
        warped, metrics = align(img, template)
        print(f"align {sheet.name}: matches={metrics['match_count']} rotation={metrics['rotation_deg']} "
              f"scale={metrics['scale']} translation=({metrics['translation_x_px']},"
              f"{metrics['translation_y_px']}) reason={metrics['reason'] or 'ok'}")
        alignment_rows.append({"sheet_file": sheet.name, **metrics,
                               "status": "aligned" if warped is not None else "skipped"})
        if warped is None:
            manual_rows.append({"sheet_file": sheet.name, "reason": metrics["reason"]})
        else:
            pages.append((sheet, warped))
    if template is None:
        print("WARNING: no --template given; scans are cut without alignment and must match "
              "the regions exactly. Check crops by eye.")
    append_csv(out_dir / "alignment_report.csv",
               ["sheet_file", "match_count", "rotation_deg", "scale",
                "translation_x_px", "translation_y_px", "status", "reason"], alignment_rows)
    append_csv(out_dir / "needs_manual_check.csv", ["sheet_file", "reason"], manual_rows)

    if not pages:
        print(f"No sheets aligned; see {out_dir / 'needs_manual_check.csv'}")
        return

    items = [(sheet, qid, cut(img, r)) for sheet, img in pages for qid, r in regions.items()]
    random.SystemRandom().shuffle(items)

    crops_dir.mkdir(parents=True, exist_ok=True)
    crop_rows, key_rows, label_rows, warnings = [], [], [], []
    for n, (sheet, qid, crop) in enumerate(items, next_n):
        crop_id = f"crop_{n:03d}"
        ok, buf = cv2.imencode(".png", crop)
        if not ok:
            sys.exit(f"Could not encode {crop_id}")
        buf.tofile(str(crops_dir / f"{crop_id}.png"))
        crop_rows.append({"crop_id": crop_id, "question_id": qid,
                          "image_path": f"crops/{crop_id}.png"})
        key_rows.append({"crop_id": crop_id, "sheet_file": sheet.name, "question_id": qid})
        label_rows.append({"crop_id": crop_id, "ground_truth_text": ""})
        warnings += [f"WARNING {crop_id} ({qid}, {sheet.name}): {w}" for w in check(crop)]

    append_csv(crops_csv, ["crop_id", "question_id", "image_path"], crop_rows)
    append_csv(labels_csv, ["crop_id", "ground_truth_text"], label_rows)
    append_csv(args.key, ["crop_id", "sheet_file", "question_id"], key_rows)

    print(f"\nCut {len(crop_rows)} crops from {len(pages)} sheets (page {args.page}) into {crops_dir}")
    print(f"crops.csv:  {crops_csv}")
    print(f"labels.csv: {labels_csv}  ({len(label_rows)} blank rows added)")
    print(f"key file:   {args.key.resolve()}  (keep private; do not give to the labeller)")
    print()
    print("\n".join(warnings) if warnings else "No warnings.")
    print()
    counts = Counter(r["question_id"] for r in crop_rows)
    print(f"{'question':<10}{'crops':>6}")
    for qid in regions:
        print(f"{qid:<10}{counts[qid]:>6}")
    print(f"{'TOTAL':<10}{len(crop_rows):>6}")


if __name__ == "__main__":
    main()

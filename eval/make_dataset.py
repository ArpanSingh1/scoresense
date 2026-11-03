"""Generate the sample labelled dataset: 20 answer crops, crops.csv and labels.csv.

crops.csv  (crop_id, question_id, image_path) is the answer-free list predict.py reads.
labels.csv (crop_id, ground_truth_text) is read only by score.py.

The crops are synthetic: text rendered with handwriting-style system fonts,
with per-character jitter, rotation, ruled lines, blur and noise. They exist
so the harness has something to run on; replace them with real scanned
answer crops (same CSV format) for a meaningful evaluation.

Usage:  python eval/make_dataset.py
"""
import csv
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).parent / "dataset"
CROPS = ROOT / "crops"
FONT_DIR = Path("C:/Windows/Fonts")
FONTS = ["Inkfree.ttf", "segoesc.ttf", "comic.ttf", "comici.ttf"]

# question_id -> four students' answers (some deliberately imperfect)
ANSWERS = {
    "Q1": ["photosynthesis", "Photosynthesis", "photosynthesys", "photo synthesis"],
    "Q2": ["42.5", "42.5 cm", "425", "42,5"],
    "Q3": ["Paris", "paris", "Paris, France", "Pairs"],
    "Q4": ["x = 7", "x=7", "x = -7", "7"],
    "Q5": [
        "The heart pumps blood",
        "heart pumps blood around the body",
        "It pumps blood",
        "The heart pumps oxygen",
    ],
}


def render(text: str, rng: random.Random) -> Image.Image:
    font = ImageFont.truetype(str(FONT_DIR / rng.choice(FONTS)), rng.randint(34, 46))
    ink = tuple(rng.randint(10, 60) for _ in range(2)) + (rng.randint(60, 140),)
    paper = tuple(rng.randint(238, 252) for _ in range(3))

    width = int(font.getlength(text) * 1.25) + 60
    height = font.size * 2 + 20
    img = Image.new("RGB", (width, height), paper)
    draw = ImageDraw.Draw(img)

    # faint ruled line under the writing
    line_y = int(height * 0.72)
    draw.line([(0, line_y), (width, line_y)], fill=(170, 190, 225), width=2)

    x = 25 + rng.randint(0, 10)
    base_y = line_y - font.size - 8
    for ch in text:
        if ch == " ":
            x += int(font.getlength(" ") * rng.uniform(0.8, 1.4))
            continue
        cw = int(font.getlength(ch)) + 16
        glyph = Image.new("L", (cw, font.size + 30), 0)
        ImageDraw.Draw(glyph).text((8, 4), ch, font=font, fill=255)
        glyph = glyph.rotate(rng.uniform(-8, 8), resample=Image.BICUBIC, expand=True)
        img.paste(Image.new("RGB", glyph.size, ink), (x, base_y + rng.randint(-4, 4)), glyph)
        x += int(font.getlength(ch) * rng.uniform(0.9, 1.08))

    img = img.rotate(rng.uniform(-2, 2), resample=Image.BICUBIC, fillcolor=paper)
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.8)))
    px = img.load()
    for _ in range(width * height // 40):
        i, j = rng.randrange(width), rng.randrange(height)
        v = rng.randint(-25, 25)
        px[i, j] = tuple(max(0, min(255, c + v)) for c in px[i, j])
    return img


def main() -> None:
    rng = random.Random(1234)
    CROPS.mkdir(parents=True, exist_ok=True)
    crop_rows, label_rows = [], []
    n = 0
    for qid, answers in ANSWERS.items():
        for text in answers:
            n += 1
            crop_id = f"crop_{n:03d}"
            render(text, rng).save(CROPS / f"{crop_id}.png")
            crop_rows.append({"crop_id": crop_id, "question_id": qid,
                              "image_path": f"crops/{crop_id}.png"})
            label_rows.append({"crop_id": crop_id, "ground_truth_text": text})

    for name, rows in (("crops.csv", crop_rows), ("labels.csv", label_rows)):
        with open(ROOT / name, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote {len(crop_rows)} crops to {CROPS}, plus {ROOT / 'crops.csv'} and {ROOT / 'labels.csv'}")


if __name__ == "__main__":
    main()

"""Run the whole marking pipeline on one test folder.

The folder holds one assessment for one class:
    <test>/scans/          scanned sheets: PDF (one file per student, page N = page N),
                           or PNG/JPG (single-page sheets, one file per student)
    <test>/regions.json    answer regions (regions.html)
    <test>/key.json        answer key (see mark.py; key_example.json)
    <test>/blank.png       optional: the blank sheet, to align scans to (cut_crops.py --template)
    <test>/settings.json   optional: {"rotate": 90} turns every scanned page 90 degrees
                           anticlockwise first (90, 180 or 270), for scans that come in sideways

It renders PDF pages, cuts the answer crops, reads them with the model, marks
them against the key, and writes, inside the test folder:
    work/      crops/, crops.csv, labels.csv, page images: what review.html opens
    private/   sheet_key.csv: crop -> student's file; keep it to the teacher
    results/   predictions.csv, marks.csv

Then it opens review.html for the teacher. An earlier run's output is never
overwritten: pass --redo to delete work/, private/ and results/ and start again.

Usage:  python eval/run.py TEST_FOLDER [--redo] [--no-open]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image

from model import MODEL, OLLAMA_URL

HERE = Path(__file__).parent
PDF_DPI = 150
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
OUTPUTS = ("work", "private", "results")


def fail(msg: str) -> None:
    sys.exit(f"\nSTOPPED: {msg}")


def step(n: int, title: str) -> None:
    print(f"\n=== {n}. {title} ===", flush=True)


def run(script: str, *args: str) -> None:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    if subprocess.run([sys.executable, str(HERE / script), *args], env=env).returncode != 0:
        fail(f"{script} failed (see above).")


def check_folder(test: Path) -> tuple[dict, dict, int]:
    if not test.is_dir():
        fail(f"{test} is not a folder.")
    for name in ("regions.json", "key.json"):
        if not (test / name).is_file():
            fail(f"{test} has no {name}.")
    if not (test / "scans").is_dir():
        fail(f"{test} has no scans folder. Put the scanned sheets in {test / 'scans'}.")
    regions = json.loads((test / "regions.json").read_text(encoding="utf-8"))
    key = json.loads((test / "key.json").read_text(encoding="utf-8"))
    no_key = sorted(set(regions) - set(key))
    if no_key:
        fail(f"key.json has no answer for {', '.join(no_key)} (they are in regions.json).")
    rotate = 0
    if (test / "settings.json").is_file():
        rotate = json.loads((test / "settings.json").read_text(encoding="utf-8")).get("rotate", 0)
        if rotate not in (0, 90, 180, 270):
            fail(f"settings.json rotate must be 0, 90, 180 or 270, got {rotate!r}.")
    return regions, key, rotate


def check_model() -> None:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as resp:
            names = {m["name"] for m in json.load(resp)["models"]}
    except OSError:
        fail(f"Ollama is not running at {OLLAMA_URL}. Start the Ollama app and run this again.")
    if MODEL not in names:
        fail(f"Ollama does not have {MODEL}. Run:  ollama pull {MODEL}")


def render_pages(scans: Path, pages_dir: Path, rotate: int) -> dict[int, int]:
    """Write page N of every sheet to pages_dir/page_N/<sheet name>.png; return pages per N."""
    counts: dict[int, int] = {}
    seen: dict[str, Path] = {}

    def save(img: Image.Image, page: int, stem: str) -> None:
        out = pages_dir / f"page_{page}"
        out.mkdir(parents=True, exist_ok=True)
        if rotate:
            img = img.rotate(rotate, expand=True)
        img.convert("RGB").save(out / f"{stem}.png")
        counts[page] = counts.get(page, 0) + 1

    files = sorted(p for p in scans.iterdir() if p.is_file())
    for f in files:
        ext = f.suffix.lower()
        if ext not in IMAGE_EXTS and ext != ".pdf":
            print(f"  skipped {f.name} (not a PDF, PNG or JPG)")
            continue
        if f.stem in seen:
            fail(f"{f.name} and {seen[f.stem].name} have the same name; rename one.")
        seen[f.stem] = f
        if ext == ".pdf":
            pdf = pdfium.PdfDocument(str(f))
            for i, page in enumerate(pdf, 1):
                save(page.render(scale=PDF_DPI / 72).to_pil(), i, f.stem)
            print(f"  {f.name}: {len(pdf)} pages", flush=True)
            pdf.close()
        else:
            with Image.open(f) as img:
                save(img, 1, f.stem)
            print(f"  {f.name}: 1 page")
    if not seen:
        fail(f"No PDF, PNG or JPG files in {scans}.")
    return counts


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("test", type=Path, help="the test folder (scans/, regions.json, key.json)")
    ap.add_argument("--redo", action="store_true",
                    help="delete this folder's work/, private/ and results/ and run again")
    ap.add_argument("--no-open", action="store_true", help="don't open review.html at the end")
    args = ap.parse_args()
    test = args.test.resolve()
    work, private, results = (test / d for d in OUTPUTS)

    step(1, "Checking the test folder")
    regions, key, rotate = check_folder(test)
    existing = [d for d in OUTPUTS if (test / d).exists()]
    if existing and not args.redo:
        question = (f"  This test was already run ({', '.join(existing)} exist). "
                    "Delete those results and run again? [y/N] ")
        if not (sys.stdin.isatty() and input(question).strip().lower() in ("y", "yes")):
            fail(f"{test} already has {', '.join(existing)} from an earlier run. "
                 "Run again with --redo to delete them and start over.")
    check_model()
    template = next((test / n for n in ("blank.png", "blank.jpg", "blank.jpeg") if (test / n).is_file()), None)
    print(f"  {len(regions)} answer regions, {len(key)} questions in the key, model {MODEL}")
    print(f"  alignment: {'to ' + template.name if template else 'none (no blank.png)'}"
          f"{f', pages turned {rotate} degrees' if rotate else ''}")
    for d in existing:
        shutil.rmtree(test / d)
        print(f"  deleted the earlier {d}/")

    step(2, "Reading the scans")
    counts = render_pages(test / "scans", work / "pages", rotate)

    step(3, "Cutting out the answers")
    region_pages = sorted({r["page"] for r in regions.values()})
    first = True
    for page in region_pages:
        if page not in counts:
            print(f"  page {page}: no scanned sheet has this page; its questions are skipped")
            continue
        print(f"\n  page {page} ({counts[page]} sheets)", flush=True)
        extra = ["--template", str(template)] if template else []
        run("cut_crops.py", "--sheets", str(work / "pages" / f"page_{page}"),
            "--regions", str(test / "regions.json"), "--page", str(page),
            "--out-dir", str(work), "--key", str(private / "sheet_key.csv"),
            *extra, *([] if first else ["--append"]))
        first = False
    if first:
        fail("No page with answer regions was found in the scans.")

    step(4, "Reading the answers with the model (takes a few seconds per answer)")
    results.mkdir(parents=True, exist_ok=True)
    run("predict.py", "--crops-list", str(work / "crops.csv"), "--out", str(results / "predictions.csv"))

    step(5, "Marking against the key")
    run("mark.py", "--key", str(test / "key.json"), "--predictions", str(results / "predictions.csv"),
        "--crops-list", str(work / "crops.csv"), "--sheet-key", str(private / "sheet_key.csv"),
        "--out", str(results / "marks.csv"))

    step(6, "Teacher review")
    print(f"  Open review.html, then choose:\n"
          f"    1. crops folder:  {work}\n"
          f"    2. marks.csv:     {results / 'marks.csv'}")
    if not args.no_open:
        os.startfile(HERE / "review.html")
    print("\nDone.")


if __name__ == "__main__":
    main()

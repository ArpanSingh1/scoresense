"""Compare models on what the app does: find each answer's question number (booklet.process).

    python eval/bench_booklet.py MODEL [MODEL ...]

Runs booklet.process on the pages in SETS with each model and compares the question numbers it
ends up with to the truth (written down by hand from the page images):
    right     a question found under its true number
    wrong     a number given that is not on those pages: an answer silently filed wrong
    missed    a true number not found (or shown as "?", which the teacher is told about)
    flagged   questions shown as "?" for the teacher
Numbers covered by correction fluid are left out of the truth: no model can read them.
"""
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import booklet

U = Path(r"C:\ocr_eval\try\bench\upright")
M = Path(r"C:\ocr_eval\try\bench\math")
SETS = {
    "SST": ([U / f"page_{n:02d}.png" for n in (2, 3, 4, 18, 21)],
            [str(n) for n in range(1, 9)] + ["13", "12", "9", "14", "15", "16", "17", "18", "25",
                                             "34.1", "34.2", "34.3", "35.1", "21", "22"]),
    "Maths": ([M / f"page_{n:02d}.png" for n in (3, 4, 5, 7, 18)],
              [str(n) for n in range(1, 19)] + ["22", "31"]),
}


def run(model: str) -> None:
    booklet.MODEL = model
    for name, (pages, truth) in SETS.items():
        out = Path(tempfile.mkdtemp())
        start = time.time()
        r = booklet.process(pages, out, progress=lambda m: None)
        secs = time.time() - start
        shutil.rmtree(out)
        got = [q["number"] for q in r["questions"] if q["number"] not in ("start", "?")]
        right = Counter(got) & Counter(truth)
        wrong = sorted((Counter(got) - Counter(truth)).elements(), key=lambda n: float(n))
        missed = sorted((Counter(truth) - right).elements(), key=lambda n: float(n))
        flagged = sum(q["number"] == "?" for q in r["questions"])
        print(f"{model:22} {name:6} right {sum(right.values()):2}/{len(truth)}  wrong {len(wrong)} {wrong}  "
              f"missed {missed}  '?' {flagged}  {secs / len(pages):.0f} s/page", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for m in sys.argv[1:]:
        run(m)

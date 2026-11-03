"""Read one or more images with read_answer() and print what the model saw.

For quick checks on your own images; nothing is written to disk. Images can
be files or folders (every PNG/JPG inside is read).

Usage:  python eval/try_image.py IMAGE_OR_FOLDER [...]
"""
import sys
from pathlib import Path

from model import REVIEW_THRESHOLD, read_answer

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def fmt(value) -> str:
    return "  -   " if value is None else f"{value:.4f}"


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__.strip())
    paths = []
    for arg in map(Path, sys.argv[1:]):
        if arg.is_dir():
            paths += sorted(p for p in arg.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        elif arg.is_file():
            paths.append(arg)
        else:
            sys.exit(f"Not found: {arg}")

    print(f"{'image':<24}{'conf_min':>9}{'conf_mean':>10}  review  text")
    for path in paths:
        result = read_answer(str(path), "try")
        review = "YES" if result["needs_review"] else "no"
        print(f"{path.name:<24}{fmt(result['confidence']):>9}{fmt(result['confidence_mean']):>10}"
              f"  {review:<6}  {result['text']!r}")
    print(f"\nreview = YES when conf_min < {REVIEW_THRESHOLD} (placeholder threshold) or no text.")


if __name__ == "__main__":
    main()

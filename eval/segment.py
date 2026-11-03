"""Cut an answer-booklet page into answers, using the numbers students write in the margin.

Ruled answer booklets have a red margin line on the left; students write the question number
left of it and the answer to its right. find_marks() finds the blue-pen marks left of that line
(red teacher ticks are ignored), and cut_blocks() cuts the page into blocks, one from each mark
down to the next: the text above the first mark continues the previous page's answer.

Which marks start a new question and which are numbered points inside an answer is decided
later, from the numbers the model reads (see booklet.py).
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Block:
    top: int
    bottom: int
    mark: tuple[int, int, int, int] | None  # x0, y0, x1, y1 of the margin mark; None = continuation


def ink_masks(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Blue-pen ink and red (margin line, teacher) masks of a BGR image."""
    b, g, r = (img[..., i].astype(int) for i in range(3))
    gray = (b + g + r) // 3
    # Dark blue or black pen; ruled lines are light blue. Red teacher ink is left out.
    pen = ((b - r > 30) & (gray < 150)) | ((gray < 90) & (r - b < 30))
    red = r - g > 30                             # the margin line is pink, teacher ink red
    return pen.astype(np.uint8), red.astype(np.uint8)


def margin_x(red: np.ndarray) -> int | None:
    """x of the red margin line (the left one of a double line), if one runs down the left 40%."""
    h, w = red.shape
    # Summed over 9 columns: a faint or slightly slanted line is broken up in any single column.
    cols = np.convolve(red[:, : int(w * 0.4)].sum(axis=0), np.ones(9), mode="same")
    if cols.max() < h * 0.4:
        return None
    return int(np.where(cols >= cols.max() * 0.6)[0][0])


def line_x(img: np.ndarray) -> int | None:
    """x of a long vertical line in the left 25%, of any colour: the margin line when it is printed
    too pale for margin_x (on some booklets every other page). Writing never makes a line this long
    and the ruled lines run across, so they are removed by the vertical opening."""
    h, w = img.shape[:2]
    left = img[:, int(w * 0.04): int(w * 0.25)]   # margins sit at 10-20%; tables further right
    # The pale line breaks up into pieces: keep vertical runs of 8% of the page height or more,
    # which handwriting doesn't make, and ask that they cover half the page between them.
    ink = (left.min(axis=2) < 238).astype(np.uint8)
    lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((int(h * 0.08), 1), np.uint8))
    # Summed over 5 columns, as a slightly slanted line moves across them.
    cols = np.convolve(lines.sum(axis=0), np.ones(5), mode="same")
    # A printed line is a few pixels wide; the scanner's shadow at the page edge is a wide strip.
    strong = np.concatenate([[0], (cols >= h * 0.5).astype(np.int8), [0]])
    starts, ends = np.where(np.diff(strong) == 1)[0], np.where(np.diff(strong) == -1)[0]
    thin = [a + int(np.argmax(cols[a:b])) for a, b in zip(starts, ends) if b - a <= w * 0.012]
    return int(w * 0.04) + int(thin[0]) if thin else None


def find_margin(img: np.ndarray) -> int | None:
    """x of the margin line of an upright page: by its red colour, else as a long vertical line."""
    return margin_x(ink_masks(img)[1]) or line_x(img)


def right_margin_x(red: np.ndarray) -> int | None:
    """x of a red line down the right 30% of the page: the rough-work column starts there."""
    h, w = red.shape
    x0 = int(w * 0.7)
    cols = np.convolve(red[:, x0:].sum(axis=0), np.ones(9), mode="same")
    if cols.max() < h * 0.4:
        return None
    return x0 + int(np.where(cols >= cols.max() * 0.6)[0][0])


def find_marks(img: np.ndarray, margin: int | None = None) -> tuple[int, list[tuple[int, int, int, int]]]:
    pen, red = ink_masks(img)
    h, w = pen.shape
    mx = margin or find_margin(img)
    if mx is None:
        mx = int(w * 0.12)
    # A label may start in the margin and run a little past the line ("21.(1)").
    zone = pen[:, int(w * 0.04): mx - 4]   # not the scanner's dark edge
    zone = cv2.morphologyEx(zone, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    rows = zone.sum(axis=1)
    min_ink = max(15, int(h * 0.006))
    marks, start, gap = [], None, 0
    line_gap = int(h * 0.012)                    # rows of a mark split by less than this are one mark
    for y in range(h + 1):
        on = y < h and rows[y] > 0
        if on:
            start = y if start is None else start
            gap = 0
        elif start is not None:
            gap += 1
            if gap > line_gap or y == h:
                end = y - gap
                part = zone[start:end + 1]
                if part.sum() >= min_ink and end - start >= h * 0.008:
                    xs = np.where(part.any(axis=0))[0]
                    x0 = int(w * 0.04)
                    marks.append((x0 + int(xs[0]), start, x0 + int(xs[-1]), end))
                start, gap = None, 0
    return mx, marks


def cut_blocks(img: np.ndarray, margin: int | None = None) -> tuple[int, list[Block]]:
    """Blocks of the page, top to bottom; the first is a continuation if there is ink above the first mark."""
    pen, _ = ink_masks(img)
    h = pen.shape[0]
    mx, marks = find_marks(img, margin)
    pad = int(h * 0.012)
    rx = right_margin_x(ink_masks(img)[1]) or pen.shape[1]
    # Cut each page in the gap between lines just above a mark, not at a fixed distance, which
    # can slice through the last line of the answer above (fractions and roots reach far down).
    ink = np.convolve(pen[:, int(pen.shape[1] * 0.04):rx].sum(axis=1), np.ones(5), mode="same")

    def cut_above(y0: int, floor: int) -> int:
        lo = max(floor, y0 - int(h * 0.045))
        if lo >= y0:
            return max(0, y0 - pad)
        rows = np.arange(lo, y0)
        return int(rows[np.argmin(ink[lo:y0] + (y0 - rows) * 0.01)])   # emptiest row, nearest the mark

    tops = []
    for i, m in enumerate(marks):
        tops.append(cut_above(m[1], marks[i - 1][3] + 1 if i else 0))
    blocks = []
    text_rows = np.where(pen[:, mx:rx].sum(axis=1) > 3)[0]   # rough work right of rx doesn't count
    first = tops[0] if tops else h
    if len(text_rows) and text_rows[0] < first - pad:
        blocks.append(Block(int(text_rows[0]) - pad // 2, first, None))
    for i, m in enumerate(marks):
        bottom = tops[i + 1] if i + 1 < len(marks) else h
        below = text_rows[(text_rows >= tops[i]) & (text_rows < bottom)]
        if len(below):
            bottom = min(bottom, int(below[-1]) + pad)
        blocks.append(Block(tops[i], bottom, m))
    return mx, blocks


def draw(img: np.ndarray, mx: int, blocks: list[Block]) -> np.ndarray:
    out = img.copy()
    cv2.line(out, (mx, 0), (mx, out.shape[0]), (0, 200, 0), 2)
    for i, b in enumerate(blocks):
        color = (0, 140, 255) if b.mark is None else (255, 0, 200)
        cv2.rectangle(out, (4, b.top), (out.shape[1] - 4, b.bottom), color, 3)
        if b.mark:
            x0, y0, x1, y1 = b.mark
            cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.putText(out, str(i), (out.shape[1] - 60, b.top + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    return out


if __name__ == "__main__":
    import sys
    for path in sys.argv[1:-1]:
        img = cv2.imread(path)
        mx, blocks = cut_blocks(img)
        print(path, "margin", mx, "blocks", [(b.top, b.bottom, b.mark is not None) for b in blocks])
        cv2.imwrite(f"{sys.argv[-1]}/seg_{path.replace(chr(92), '/').split('/')[-1]}", draw(img, mx, blocks))

"""Turn an uploaded answer booklet (PDF or photos) into answer crops grouped by question.

    pages:   PDF pages are rendered; each file's pages are turned upright (see orient()).
    crops:   segment.cut_blocks() cuts each page at the marks in the margin.
    numbers: the model reads the number of each mark from a small crop of the margin.
    grouping: assign() decides which marks start a question and which are numbered points
             inside an answer, and attaches text that continues onto the next page.

Every place where the number can't be trusted is flagged for the teacher, never guessed:
an unreadable or covered number, a number already used earlier in the booklet.
A number hidden under correction fluid leaves no mark at all on these scans, so that answer
stays inside the crop above it; the teacher sees two answers in one crop.
"""
import base64
import json
import math
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import pypdfium2 as pdfium

from model import MODEL, OLLAMA_URL
from segment import Block, cut_blocks, ink_masks, margin_x, right_margin_x

PDF_DPI = 150
NUMBER_PROMPT = (
    "This is the left margin of a student's exam answer booklet, with the start of one answer. "
    "What question number did the student write in the margin? Copy it exactly as written, "
    "for example 13, 34.1, 25 b) or (2). If what is written is not a number (a sign such as => "
    "or a word), answer -. If a number is covered, crossed out or unreadable, answer ?. "
    "Never guess. Return only the number.")
NUMBER_CONFIDENCE = 0.5
CIRCLED = {chr(0x2460 + i): f"({i + 1})" for i in range(20)}
# A handwritten Q is often read as 0 ("016." for "Q16."), so up to three digits, leading zero dropped.
LABEL = re.compile(r"^\s*(?:Q\.?|A[nmu]s\.?)?\s*(\(?)\s*(\d{1,3})(?:\s*\.\s*(\d))?")


@dataclass
class Crop:
    page: int
    file: str             # crop image name inside the job's crops/ folder
    label: str            # number the model read for this crop's mark ("" for a continuation)
    confidence: Optional[float]
    kind: str = ""        # question, point, continuation, note


@dataclass
class Question:
    number: str           # "13", "34.1", or "?" when it couldn't be read
    crops: list[Crop] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    unsure: bool = False  # its number was read without confidence


def load_pages(files: list[Path]) -> list[tuple[str, list[np.ndarray]]]:
    """(file name, its pages as BGR images) for each uploaded file, in upload order."""
    out = []
    for f in files:
        if f.suffix.lower() == ".pdf":
            pdf = pdfium.PdfDocument(str(f))
            pages = [cv2.cvtColor(np.array(p.render(scale=PDF_DPI / 72).to_pil().convert("RGB")),
                                  cv2.COLOR_RGB2BGR) for p in pdf]
            pdf.close()
        else:
            img = cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_COLOR)
            pages = [img] if img is not None else []
        out.append((f.name, pages))
    return out


def ask(image: np.ndarray, prompt: str) -> tuple[str, Optional[float], Optional[float]]:
    """Model's answer, minimum probability of its digit tokens (of all tokens if it has no digits),
    and mean token probability."""
    ok, png = cv2.imencode(".png", image)
    body = {"model": MODEL, "stream": False, "logprobs": True,
            "options": {"temperature": 0, "num_predict": 20},
            "messages": [{"role": "user", "content": prompt, "images": [base64.b64encode(png).decode()]}]}
    if MODEL.startswith("qwen3"):
        body["think"] = False                    # a short answer is wanted, not reasoning
    req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.load(resp)
    tokens = [t for t in data.get("logprobs") or [] if t.get("logprob") is not None]
    lp = [t["logprob"] for t in tokens]
    # For a question number, how sure the model is of its digits: "25 b)" is unsure about "b)".
    digits = [t["logprob"] for t in tokens if any(ch.isdigit() for ch in t.get("token", ""))] or lp
    text = data["message"]["content"].strip().strip("`\"' ")
    return text, (math.exp(min(digits)) if digits else None), (math.exp(sum(lp) / len(lp)) if lp else None)


ROTATIONS = {0: None, 90: cv2.ROTATE_90_COUNTERCLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_CLOCKWISE}


def rotate(img: np.ndarray, degrees: int) -> np.ndarray:
    return img if not degrees else cv2.rotate(img, ROTATIONS[degrees])


def orient(pages: list[np.ndarray]) -> int:
    """Rotation (anticlockwise degrees) that makes this file's pages upright.

    The margin line must run down the page: that leaves two candidates, 180 degrees apart.
    The model reads a line of handwriting both ways round and the more confident reading wins.
    Scanned booklets come through one way round, so one page decides for the whole file."""
    for img in pages:
        upright = [d for d in ROTATIONS if margin_x(ink_masks(rotate(img, d))[1]) is not None]
        if not upright:
            continue
        a = upright[0] % 180
        # The margin line is on the left: when only one way round has it there, that decides.
        candidates = [d for d in (a, a + 180) if d in upright]
        if len(candidates) == 1:
            return candidates[0]
        test = rotate(img, a)
        pen, _ = ink_masks(test)
        rows = np.where(pen.sum(axis=1) > pen.shape[1] * 0.02)[0]
        if not len(rows):
            continue
        y = int(np.median(rows))
        h = test.shape[0]
        strip = test[max(0, y - h // 30): y + h // 30]
        means = []
        for s in (strip, cv2.rotate(strip, cv2.ROTATE_180)):
            _, _, mean = ask(s, "Transcribe this line of handwriting. Return only the text.")
            means.append(mean or 0)
        return candidates[int(means[1] > means[0])]
    return 0


def mark_crop(img: np.ndarray, mx: int, b: Block) -> np.ndarray:
    h, w = img.shape[:2]
    _, y0, _, y1 = b.mark
    pad = int(h * 0.015)
    return img[max(0, y0 - pad): y1 + pad, int(w * 0.01): min(w, mx + int(w * 0.01))]


def answer_crop(img: np.ndarray, b: Block) -> np.ndarray:
    """The answer, without the rough-work column right of a right-hand margin line."""
    w = img.shape[1]
    rx = right_margin_x(ink_masks(img)[1])
    return img[max(0, b.top): b.bottom, int(w * 0.01): rx - 2 if rx else int(w * 0.99)]


def parse(label: str) -> tuple[Optional[str], bool]:
    """(question number such as "13" or "34.1", written in brackets) from what the model read."""
    label = "".join(CIRCLED.get(c, c) for c in label)
    m = LABEL.match(label)
    if not m:
        return None, False
    number = str(int(m.group(2))) + (f".{m.group(3)}" if m.group(3) else "")
    return number, bool(m.group(1))


def same_number(a: Optional[str], b: Optional[str]) -> bool:
    """Equal, or equal but for a dot: "Ans 31." is sometimes read as 3.1."""
    return bool(a and b) and (a == b or a.replace(".", "") == b.replace(".", ""))


def sure(label: str, conf: Optional[float]) -> bool:
    return label.strip() not in ("", "?") and conf is not None and conf >= NUMBER_CONFIDENCE


def merge_pairs(blocks: list[Block], reads: list[tuple[str, Optional[float]]], h: int
                ) -> tuple[list[Block], list[tuple[str, Optional[float]]]]:
    """Join a "Q1." line with the "Ans 1." line under it: two marks a line or two apart that
    give the same number, or where one reads a number and the other can't be read."""
    out_b, out_r = [blocks[0]] if blocks else [], [reads[0]] if reads else []
    for b, r in zip(blocks[1:], reads[1:]):
        a, ra = out_b[-1], out_r[-1]
        if a.mark and b.mark and b.mark[1] - a.mark[3] < h * 0.06:
            na, nb = parse(ra[0])[0], parse(r[0])[0]
            sa, sb = sure(*ra), sure(*r)
            if same_number(na, nb) or (sa and not sb and nb in (None, na)) or (sb and not sa and na in (None, nb)):
                out_b[-1] = Block(a.top, b.bottom, a.mark)
                out_r[-1] = max(ra, r, key=lambda x: (sure(*x), x[1] or 0))
                continue
        out_b.append(b)
        out_r.append(r)
    return out_b, out_r


def assign(crops: list[Crop]) -> list[Question]:
    """Group crops (in booklet order) into questions; see the module docstring.

    A number the model is sure of is taken as read. One it is unsure of is taken only when it is
    the next question (or the one after), the commonest case; otherwise the mark stays with the
    answer above it, and that answer is flagged for the teacher to check. A number that jumps
    backwards or far ahead is flagged even when the model is sure: on the test booklet, half-covered
    numbers were read with confidence as other numbers (26 as 21, 37 as 34)."""
    questions: list[Question] = []
    seen: dict[str, int] = {}
    current: Optional[Question] = None
    last_point = 0
    for c in crops:
        if c.kind == "continuation":
            if current is None:                  # a heading such as "Section A", or a cover page
                current = Question("start")
                questions.append(current)
            current.crops.append(c)
            continue
        number, bracketed = parse(c.label)
        unsure = not sure(c.label, c.confidence)
        main = int(number.split(".")[0]) if number else None
        cur = current.number if current else "start"
        cur_main = int(cur.split(".")[0]) if cur not in ("?", "start") else None
        # "35" read as 3.5: take the number without the dot when that is the next question.
        if number and "." in number and cur_main is not None and main not in (cur_main, cur_main + 1)                 and int(number.replace(".", "")) in (cur_main + 1, cur_main + 2):
            number = number.replace(".", "")
            main = int(number)
        if same_number(number, cur):             # the same number again: "Ans 6." after "Q6.", or rewritten
            if not (current.unsure and not unsure):
                c.kind = "question"
                current.crops.append(c)
                continue
            # The earlier, unsure reading was probably a different number ("1." read as 2 before
            # a clear "2."): that answer becomes "?" for the teacher, and this one starts the question.
            if seen.get(current.number) == current.crops[0].page:
                del seen[current.number]
            first = next((x for x in current.crops if x.kind == "question"), current.crops[0])
            current.number, current.unsure = "?", False
            current.flags = [f"number not readable (read as \"{first.label}\")"]
            cur, cur_main = "?", None
        if cur_main is None:
            # No number to follow on from (the sheet may start at any question): take what is
            # read, and tell the teacher when the model was unsure of it.
            plausible = main is not None
        else:
            plausible = main is not None and (main in (cur_main + 1, cur_main + 2)
                                              or (main == cur_main and "." in number))
        # A point inside the current answer: written in brackets, or a small number below the
        # question's own that carries on its 1, 2, 3.
        if cur_main is not None and number and not unsure and (
                bracketed or (main <= 9 and "." not in number and main < cur_main
                              and last_point <= main <= last_point + 2)):
            c.kind = "point"
            last_point = main
            current.crops.append(c)
            continue
        not_number = c.label.strip() in ("-", "–") and not unsure
        if current is not None and (number is None or (unsure and not plausible)):
            c.kind = "note"                      # a word or sign (=>) in the margin, or unsure
            if unsure and not not_number:
                flag = "a mark in the margin could not be read; check this is all one answer"
                if flag not in current.flags:
                    current.flags.append(flag)
            current.crops.append(c)
            continue
        c.kind = "question"
        q = Question(number if number and (plausible or not unsure) else "?")
        if q.number == "?":
            q.flags.append(f"number not readable (read as \"{c.label or 'nothing'}\")")
        elif q.number in seen:
            q.flags.append(f"Q{q.number} was already answered on page {seen[q.number]}")
        else:
            seen[q.number] = c.page
        if unsure and cur_main is None and q.number != "?":
            q.flags.append(f"number read as \"{c.label}\", but not for sure: check it")
            q.unsure = True
        if cur_main is not None and q.number != "?" and not plausible and (main < cur_main or main > cur_main + 5):
            q.flags.append(f"Q{q.number} comes after Q{cur}: check the number (it may be answered out of order)")
        questions.append(q)
        q.crops.append(c)
        current, last_point = q, 0
    return questions


def build_result(crops: list[Crop], pages: int, skipped: list[int]) -> dict:
    return {"pages": pages, "pages_without_margin": skipped,
            "questions": [{"number": q.number, "flags": q.flags,
                           "crops": [c.__dict__ for c in q.crops]} for q in assign(crops)]}


def process(files: list[Path], out_dir: Path, progress: Callable[[str], None] = print,
            partial: Callable[[dict], None] = lambda r: None) -> dict:
    """Write crops to out_dir/crops and return the result (also saved as out_dir/result.json).

    partial() gets the result so far after each page, so the answers can be shown as they come."""
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    crops: list[Crop] = []
    page_no = 0
    skipped = []
    loaded = load_pages(files)
    total = sum(len(p) for _, p in loaded)
    for name, pages in loaded:
        progress(f"Turning the pages of {name} upright")
        degrees = orient(pages)
        for img in pages:
            page_no += 1
            img = rotate(img, degrees)
            progress(f"Page {page_no} of {total}: finding the answers")
            mx_found = margin_x(ink_masks(img)[1])
            if mx_found is None:
                skipped.append(page_no)
                cv2.imwrite(str(crops_dir / f"p{page_no:02d}_page.jpg"), img)
                crops.append(Crop(page_no, f"p{page_no:02d}_page.jpg", "", None, "continuation"))
                partial(build_result([Crop(**c.__dict__) for c in crops], total, skipped))
                continue
            mx, blocks = cut_blocks(img)
            reads: list[tuple[str, Optional[float]]] = []
            for i, b in enumerate(blocks):
                if b.mark is None:
                    reads.append(("", None))
                    continue
                progress(f"Page {page_no} of {total}: reading question number {i + 1} of {len(blocks)}")
                label, conf, _ = ask(mark_crop(img, mx, b), NUMBER_PROMPT)
                reads.append((label, conf))
            blocks, reads = merge_pairs(blocks, reads, img.shape[0])
            for i, (b, (label, conf)) in enumerate(zip(blocks, reads)):
                if b.bottom - b.top < img.shape[0] * 0.03 and b.mark is None:
                    continue
                name_i = f"p{page_no:02d}_{i:02d}.jpg"
                cv2.imwrite(str(crops_dir / name_i), answer_crop(img, b), [cv2.IMWRITE_JPEG_QUALITY, 88])
                crops.append(Crop(page_no, name_i, label, conf, "continuation" if b.mark is None else ""))
            partial(build_result([Crop(**c.__dict__) for c in crops], total, skipped))
    result = build_result(crops, total, skipped)
    (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


if __name__ == "__main__":
    import sys
    r = process([Path(p) for p in sys.argv[2:]], Path(sys.argv[1]))
    for q in r["questions"]:
        parts = ", ".join(f"p{c['page']} {c['kind']} '{c['label']}'" for c in q["crops"])
        print(f"Q{q['number']:5} {'; '.join(q['flags']):45} {parts}")

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
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import pypdfium2 as pdfium

from model import MODEL, OLLAMA_URL, READER_OPTIONS
from segment import Block, cut_blocks, find_margin, ink_masks, margin_x, right_margin_x, ruled

PDF_DPI = 150
NUMBER_PROMPT = (
    "This is the left margin of a student's exam answer booklet, with the start of one answer. "
    "What question number did the student write in the margin? Copy it exactly as written, "
    "for example 13, 34.1, 25 b) or (2). Sub-parts written with letters or Roman numerals, such as "
    "i), ii), iii), (a) or b), are not numbers: copy them as written, for example ii). "
    "If what is written is not a number (a sign such as => "
    "or a word), answer -. If a number is covered, crossed out or unreadable, answer ?. "
    "Never guess. Return only the number.")
NUMBER_CONFIDENCE = 0.5
CIRCLED = {chr(0x2460 + i): f"({i + 1})" for i in range(20)}
# A handwritten Q is often read as 0 ("016." for "Q16."), so up to three digits, leading zero dropped.
LABEL = re.compile(r"^\s*(?:Q\.?|A[nmu]s\.?)?\s*(\(?)\s*(\d{1,3})(?!\d)(?:\s*\.\s*(\d))?")
# "Q.25)" is often read as "8.25", "9.25" or "0.25": a lone 8, 9 or 0 and a dot before the number.
Q_AS_DIGIT = re.compile(r"^\s*\(?\s*[890]\s*(?:[.,]\s*(\d{1,2})|(\d{2}))(?!\d)")


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
            "options": {**READER_OPTIONS, "num_predict": 20},
            "messages": [{"role": "user", "content": prompt, "images": [base64.b64encode(png).decode()]}]}
    if MODEL.startswith("qwen3"):
        body["think"] = False                    # a short answer is wanted, not reasoning
    req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    for attempt in (1, 2):                   # the model can stall once; one stall shouldn't fail the booklet
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.load(resp)
            break
        except (TimeoutError, urllib.error.URLError):     # a stall, or the model restarting
            if attempt == 2:
                raise
            time.sleep(5)
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
    The model reads three lines of handwriting both ways round and the more confident readings win.
    Which side the margin line is found on doesn't decide it: booklets ruled on both sides have one
    line found only upside down (English booklet: the right-hand line, with the left one missed).
    Scanned booklets come through one way round, so one page decides for the whole file."""
    for img in pages:
        upright = [d for d in ROTATIONS if margin_x(ink_masks(rotate(img, d))[1]) is not None]
        if not upright:
            continue
        a = upright[0] % 180
        test = rotate(img, a)
        pen, _ = ink_masks(test)
        rows = np.where(pen.sum(axis=1) > pen.shape[1] * 0.02)[0]
        if not len(rows):
            continue
        h = test.shape[0]
        score = [0.0, 0.0]
        for q in (25, 50, 75):
            y = int(np.percentile(rows, q))
            strip = test[max(0, y - h // 30): y + h // 30]
            for k, s in enumerate((strip, cv2.rotate(strip, cv2.ROTATE_180))):
                score[k] += ask(s, "Transcribe this line of handwriting. Return only the text.")[2] or 0
        return (a, a + 180)[int(score[1] > score[0])]
    return 0


def mark_crop(img: np.ndarray, mx: int, b: Block) -> np.ndarray:
    """The margin mark, with the rest of the number when it runs across the margin line ("Q.9)"
    written over the line), up to the first gap: the space before the answer."""
    h, w = img.shape[:2]
    _, y0, _, y1 = b.mark
    # A mark taller than a line holds two ("Q.33)" with "(a)" under it): the number is the first.
    y1 = min(y1, y0 + int(h * 0.05))
    pad = int(h * 0.015)
    pen, _ = ink_masks(img[y0:y1 + 1])
    cols = pen.any(axis=0)
    right, gap, x = mx + int(w * 0.01), 0, mx + 3
    while x < min(w, mx + int(w * 0.08)):
        if cols[x]:
            right, gap = x + 2, 0
        else:
            gap += 1
            if gap > int(w * 0.008):
                break
        x += 1
    return img[max(0, y0 - pad): y1 + pad, int(w * 0.01): min(w, max(right, mx + int(w * 0.01)))]


def answer_crop(img: np.ndarray, b: Block) -> np.ndarray:
    """The answer, without the rough-work column right of a right-hand margin line."""
    w = img.shape[1]
    rx = right_margin_x(ink_masks(img)[1])
    return img[max(0, b.top): b.bottom, int(w * 0.01): rx - 2 if rx else int(w * 0.99)]


def q_reading(label: str) -> tuple[Optional[str], bool]:
    """The number if the label's first digit is a misread "Q.": ("25", True) for "8.25", where two
    digits after the dot can't be a part number; ("1", False) for "8.1", which might be part 8.1."""
    m = Q_AS_DIGIT.match(label)
    if not m:
        return None, False
    if m.group(2):                               # "829": Q.29 with the dot missed
        return str(int(m.group(2))), False
    return str(int(m.group(1))), len(m.group(1)) == 2


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
    give the same number, or where one reads a number and the other has no digits at all ("Q",
    "Ans"). An unreadable mark with digits ("?.4)") is a question of its own: MCQs are a line apart."""
    def number(label: str) -> Optional[str]:
        alt, certain = q_reading(label)
        return alt if certain else parse(label)[0]

    def no_digits(label: str) -> bool:
        """A word such as "Ans" or "Q" with no number: the other half of a "Q1. / Ans 1." pair. A
        bare "?" is a number the model couldn't read, so a question of its own."""
        return not any(ch.isdigit() for ch in label) and any(ch.isalpha() for ch in label)

    out_b, out_r = [blocks[0]] if blocks else [], [reads[0]] if reads else []
    for b, r in zip(blocks[1:], reads[1:]):
        a, ra = out_b[-1], out_r[-1]
        if a.mark and b.mark and b.mark[1] - a.mark[3] < h * 0.06:
            na, nb = number(ra[0]), number(r[0])
            sa, sb = sure(*ra), sure(*r)
            if same_number(na, nb) or (sa and not sb and no_digits(r[0])) or (sb and not sa and no_digits(ra[0])):
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
    last_good: Optional[int] = None   # the last question number that followed the order

    def clear_numbers(c: Crop) -> set[int]:
        """The question numbers a mark may be ("8.3": 3, or part 8.3). Used only to check for the
        number two ahead, so an unsure reading that agrees with the order is enough."""
        if c.kind == "continuation":
            return set()
        alt, _ = q_reading(c.label)
        n = parse(c.label)[0]
        return {int(x) for x in (alt, n) if x and "." not in x}

    for idx, c in enumerate(crops):
        if c.kind == "continuation":
            if current is None:                  # a heading such as "Section A", or a cover page
                current = Question("start")
                questions.append(current)
            current.crops.append(c)
            continue
        number, bracketed = parse(c.label)
        unsure = not sure(c.label, c.confidence)
        if bracketed and unsure and current is not None:
            # "(2)" read without confidence: a part label such as (v) or (क) in the margin (Hindi
            # booklets), not a question number, which students write as "2." or "Q2".
            c.kind = "note"
            current.crops.append(c)
            continue
        cur = current.number if current else "start"
        cur_main = int(cur.split(".")[0]) if cur not in ("?", "start") else None
        # The order is judged from the last number that followed it: a misread one ("ii)" read as
        # 10) must not make the next real number (33) look out of place.
        order = last_good if last_good is not None and current is not None and current.flags else cur_main
        alt, certain = q_reading(c.label)
        if alt and (certain or int(alt) in ((order or 0) + 1, (order or 0) + 2)):
            number = alt                         # "8.25" -> 25; "8.9" -> 9 when 9 comes next
        main = int(number.split(".")[0]) if number else None
        # "35" read as 3.5: take the number without the dot when that is the next question.
        if number and "." in number and cur_main is not None and main not in (cur_main, cur_main + 1)                 and int(number.replace(".", "")) in (cur_main + 1, cur_main + 2):
            number = number.replace(".", "")
            main = int(number)
        # The same number again after an answer that skipped one (21, then "22" read as 23, then a
        # clear 23 further on): the first was the skipped number. It takes that number, flagged.
        prev_main = next((int(x.number.split(".")[0]) for x in reversed(questions[:-1])
                          if x.number not in ("?", "start")), None)
        if (same_number(number, cur) and cur_main is not None and prev_main is not None
                and cur_main == prev_main + 2 and "." not in cur and str(prev_main + 1) not in seen
                and len(current.crops) >= 2):
            skipped = str(prev_main + 1)
            if seen.get(current.number) == current.crops[0].page:
                del seen[current.number]
            seen[skipped] = current.crops[0].page
            current.flags.append(f"read as Q{current.number}, but Q{current.number} starts again on page "
                                 f"{c.page}: taken as Q{skipped} from the order, check it")
            current.number, last_good = skipped, prev_main + 1
            cur, cur_main, order = skipped, prev_main + 1, prev_main + 1
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
                                              or (main == cur_main and "." in number)
                                              or (order is not None and main in (order + 1, order + 2)))
        # A point inside the current answer: written in brackets, or a small number below the
        # question's own that carries on its 1, 2, 3.
        next_up = bracketed and main is not None and cur_main is not None and main == cur_main + 1 and any(
            cur_main + 2 in clear_numbers(x) for x in crops[idx + 1:idx + 3])
        if cur_main is not None and number and not unsure and not next_up and (
                bracketed or (main <= 9 and "." not in number and main < cur_main
                              and last_point <= main <= last_point + 2)):
            c.kind = "point"
            last_point = main
            current.crops.append(c)
            continue
        # A number the model couldn't read ("?") between Q1 and a clear Q3 on the next mark is Q2.
        if number is None and cur_main is not None and "?" in c.label:
            nxt = next((n for n in map(clear_numbers, crops[idx + 1:idx + 3]) if n), set())
            if cur_main + 2 in nxt:
                number, main, unsure, plausible = str(cur_main + 1), cur_main + 1, True, True
                c.kind = "question"
                q = Question(number, flags=[f"number not readable; taken as Q{number} from the order: check it"],
                             unsure=True)
                seen.setdefault(number, c.page)
                questions.append(q)
                q.crops.append(c)
                current, last_point = q, 0
                continue
        not_number = c.label.strip() in ("-", "–") and not unsure
        # An unsure two-digit mark one digit off the next question and far from the order ("24" read as 94):
        # that question, flagged. One close by ("25" after 22) may be working: it is flagged below instead.
        nxt = str(cur_main + 1) if cur_main is not None else ""
        if (unsure and not plausible and number and number.isdigit() and len(number) == 2 == len(nxt)
                and not cur_main < int(number) <= cur_main + 3
                and sum(a != b for a, b in zip(number, nxt)) == 1 and nxt not in seen
                and re.fullmatch(r"\s*(?:Q\.?\s*)?\d{2}\s*[.)]?\s*", c.label)
                and not any(int(nxt) in clear_numbers(x) for x in crops[idx + 1:idx + 4])):
            c.kind = "question"
            q = Question(nxt, flags=[f"number read as \"{c.label}\", not for sure: taken as Q{nxt} from the order, "
                                     f"check it"], unsure=True)
            seen[nxt] = c.page
            last_good = int(nxt)
            questions.append(q)
            q.crops.append(c)
            current, last_point = q, 0
            continue
        if current is not None and (number is None or (unsure and not plausible)):
            c.kind = "note"                      # a word or sign (=>) in the margin, or unsure
            # Flag only a mark that might be a missed question number: it has digits and could
            # come next. Arrows, (a), (ii) and the 2s of maths working stay quietly with the answer.
            maybe_next = main is not None and cur_main is not None and cur_main < main <= cur_main + 3
            if unsure and not not_number and maybe_next:
                flag = "a mark in the margin could not be read; check this is all one answer"
                if flag not in current.flags:
                    current.flags.append(flag)
            current.crops.append(c)
            continue
        ref = last_good if last_good is not None else cur_main
        # A small number already answered, well after it: "(b)" read as 6, "ii)" as 11, after Q6 and Q11.
        # A bigger one (21 at the end, after Q36) is the student answering a skipped section again.
        if current is not None and number in seen and ref is not None and main < ref - 5 and main <= 12:
            c.kind = "note"
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
        # Out of order: it follows neither of the last two answers, and is far from both the order and
        # the answer before it. A student who skipped a section (18, 25, 26, 27 ... 36, 21) gets a flag
        # on the jump, not on every answer after it.
        out = lambda r: r is None or (not (r < main <= r + 2) and (main < r or main > r + 5))
        follows = any(r is not None and r < main <= r + 2 for r in (cur_main, prev_main))
        if (ref is not None and q.number != "?" and not follows and out(ref) and out(cur_main)
                and not q.flags):
            q.flags.append(f"Q{q.number} comes after Q{cur_main if cur_main is not None else ref}: "
                           f"check the number (it may be answered out of order)")
        elif q.number != "?":
            last_good = main
        questions.append(q)
        q.crops.append(c)
        current, last_point = q, 0
    return questions


def fit_to_paper(questions: list[dict], paper: set[int]) -> None:
    """A number that is not in the paper ("31" read as 71) takes the one unanswered number that fits
    between its neighbours (30 and 33 -> 31), flagged; with none or several, it is only flagged."""
    main = lambda q: int(q["number"].split(".")[0]) if q["number"].split(".")[0].isdigit() else None
    answered = {main(q) for q in questions}
    for i, q in enumerate(questions):
        m = main(q)
        if m is None or m in paper:
            continue
        before = next((main(x) for x in reversed(questions[:i]) if main(x) in paper), 0)
        after = next((main(x) for x in questions[i + 1:] if main(x) in paper), max(paper) + 1)
        fits = [n for n in sorted(paper) if before < n < after and n not in answered]
        if len(fits) > 1:                        # 30, "71", 33: 31 and 32 fit, 31 is one digit off
            fits = [n for n in fits if len(str(n)) == len(str(m))
                    and sum(a != b for a, b in zip(str(n), str(m))) == 1]
        q["flags"] = [f for f in q["flags"] if "comes after" not in f]
        if len(fits) == 1:
            q["flags"].append(f"read as Q{q['number']}, which is not in the paper: taken as Q{fits[0]} "
                              f"from the order, check it")
            q["number"] = str(fits[0])
            answered.add(fits[0])
        else:
            q["flags"].append(f"Q{q['number']} is not in the paper: check the number")
    # A number in the paper but in a one-question gap (25, "21", 27, with 26 nowhere): the half-covered
    # 26 read as 21. Taken when the gap's number is one digit off what was read.
    one_off = lambda a, b: len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == 1
    numbered = [i for i, q in enumerate(questions) if main(q) is not None]
    for k in range(1, len(numbered) - 1):
        q = questions[numbered[k]]
        m, p, x = main(q), main(questions[numbered[k - 1]]), main(questions[numbered[k + 1]])
        if "." in q["number"] or x != p + 2 or p < m < x:
            continue
        gap = p + 1
        if gap in answered or gap not in paper or not one_off(str(m), str(gap)):
            continue
        q["flags"] = [f for f in q["flags"] if "comes after" not in f and "already answered" not in f]
        q["flags"].append(f"read as Q{q['number']}, between Q{p} and Q{x}: taken as Q{gap} from the order, "
                          f"check it")
        page = q["crops"][0]["page"] if q.get("crops") else None
        for other in questions:
            other["flags"] = [f for f in other["flags"]
                              if f != f"Q{q['number']} was already answered on page {page}"]
        q["number"] = str(gap)
        answered.add(gap)


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
        # Booklet pages alternate between two layouts (left and right pages): when a page's margin
        # line is too faint to find, the last page of the same kind tells where it is.
        last_margin: dict[int, int] = {}
        for img in pages:
            page_no += 1
            img = rotate(img, degrees)
            progress(f"Page {page_no} of {total}: finding the answers")
            mx_found = find_margin(img)
            if mx_found is None and page_no % 2 in last_margin and ruled(img):
                mx_found = last_margin[page_no % 2]
            if mx_found is not None:
                last_margin[page_no % 2] = mx_found
            if mx_found is None:
                skipped.append(page_no)
                cv2.imwrite(str(crops_dir / f"p{page_no:02d}_page.jpg"), img)
                crops.append(Crop(page_no, f"p{page_no:02d}_page.jpg", "", None, "continuation"))
                partial(build_result([Crop(**c.__dict__) for c in crops], total, skipped))
                continue
            mx, blocks = cut_blocks(img, mx_found)
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

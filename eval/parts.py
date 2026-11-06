"""Split a question answered in parts into one item per part: Hindi papers ask 1 (i)-(v), 3 (क)-(ङ).

A question has parts when the marks scheme lists them ("1.1-1.5:1"). Its answer is transcribed (all its
crops, in order) and the text is cut at the part labels the student wrote at the start of a line:
(i), ii), (क), (ङ). Each part becomes an item of its own, "1.1" to "1.5", with the question's crops and
its own transcript, so marking, totals and the rank list work as for any question. Students often
write both labels, "(i) (क)": a label that names the same part again is skipped, and a label that
goes backwards is part of the answer, not a new part.

An MCQ part (one with a key, "7.3": "D") is checked from its text: the option it starts with, by the
same rules as a whole MCQ (model.check_option). A question whose labels can't be found stays whole,
flagged for the teacher.
"""
import re
from pathlib import Path
from typing import Callable, Optional

from model import LETTER_OPTIONS, check_option, read_answer

NL = "\n"
ROMAN = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]
DEVANAGARI = ["क", "ख", "ग", "घ", "ङ", "च"]
# How the reading model writes handwritten labels: (ङ) as (ड़), (ड.) or (इ.); (घ) as (ध); (ग) as (वा);
# (ख) as (ब). Inside brackets at the start of a line they can't be anything else.
LETTER_FORMS = {"ड़": "ङ", "ड़": "ङ", "ड": "ङ", "इ": "ङ", "ध": "घ", "वा": "ग", "ब": "ख"}
LABEL = re.compile(r"\(?\s*(x|ix|viii|vii|vi|iv|v|iii|ii|i|क|ख|ग|घ|ङ|ड़|ड़|ड|इ|ध|वा|ब)\s*\.?\s*\)\s*")
# What comes before the labels on a line: the question number "1.", "9.", "2→", "8 ->".
LINE_START = re.compile(r"^\s*(?:Q\.?\s*)?(?:\d{1,2}\s*[.):→>-]*\s*)?")
LEAD_NUMBER = re.compile(r"^\s*(?:Q\.?\s*|Ans\.?\s*)?(\d{1,2})\s*[.)]")   # "28. Three ...", "Q37) (b)"


def OWN_START(number: str) -> re.Pattern:
    """A line that starts the question itself: "5. (क)", "Q5 (i)"."""
    return re.compile(rf"^\s*(?:Q\.?\s*)?{number}\s*[.):→>-]*\s*" + LABEL.pattern)


def label_index(label: str) -> Optional[int]:
    """(iii) -> 3, (ग) -> 3."""
    label = LETTER_FORMS.get(label, label)
    if label in ROMAN:
        return ROMAN.index(label) + 1
    if label in DEVANAGARI:
        return DEVANAGARI.index(label) + 1
    return None


def split_text(text: str, restart_ends: bool = False) -> list[tuple[int, str]]:
    """[(part index, its text)] in order, from a transcript with part labels at the start of lines.
    With restart_ends, labels starting again at the first part ("(i)" after "(ग)") end the question: the
    rest is returned as part 0, the next question's answer written without its number being found."""
    parts: list[tuple[int, list[str]]] = []
    current = 0
    lines = text.splitlines()
    for n, line in enumerate(lines):
        body, labels = line, []
        while True:                               # "(i) (क)", "(i) 9.(क)": labels, maybe with the number
            rest = body[LINE_START.match(body).end():]
            m = LABEL.match(rest)
            if not m or label_index(m.group(1)) is None:
                break
            labels.append(label_index(m.group(1)))
            body = rest[m.end():]
        if restart_ends and labels and labels[0] == 1 and current >= 2:
            return [(i, NL.join(ls).strip()) for i, ls in parts] + [(0, NL.join(lines[n:]))]
        new = next((i for i in labels if i > current), None)
        if new is not None:
            parts.append((new, [body]))
            current = new
        elif parts:
            parts[-1][1].append(line)            # the answer goes on (or a label going backwards)
    return [(i, NL.join(ls).strip()) for i, ls in parts]


def mcq_from_text(text: str, expected: str) -> dict:
    """The option a part's text starts with, checked against the key, as mcq.check records it."""
    if not text.strip():
        return {"read": "", "option": None, "reasons": ["nothing written"], "key": expected, "verdict": "review"}
    _, option, reasons = check_option(text.strip(), LETTER_OPTIONS)
    written = list(dict.fromkeys(re.findall(r"\(\s*([A-D])\s*\)", text)))
    if len(written) > 1:                      # "(A) (C) (D)": the model copies crossed-out options too
        option, reasons = None, [f"several options written ({', '.join(written)}), maybe crossed out"]
    verdict = "review" if option is None else "right" if option == expected else "wrong"
    return {"read": text.strip(), "option": option, "reasons": reasons, "key": expected, "verdict": verdict}


def part_crop(crops: list[dict], idx: int) -> Optional[dict]:
    """The crop whose margin label is part idx ("iii)" for 3), if the part has one of its own."""
    for c in crops:
        m = LABEL.fullmatch((c.get("label") or "").strip())
        if m and label_index(m.group(1)) == idx:
            return c
    return None


def recheck_mcq(part: dict, crops_dir: Path) -> None:
    """A part read as a wrong option is read again from its own crop, the way whole MCQs are: the
    transcript keeps crossed-out options ("(A) (C) (D)" read as "(A)"), the crop check sees them."""
    m = part.get("mcq")
    if not m or m["verdict"] == "right" or crops_dir is None or m["reasons"] == ["nothing written"]:
        return
    c = part_crop(part["crops"], int(part["number"].split(".")[1]))
    if not c:
        return
    from mcq import kept_option, read_question
    if m["verdict"] == "wrong":
        again = read_question(crops_dir, [c])
        if again["option"] == m["key"]:
            m.update(option=again["option"], read=again["read"], reasons=[], verdict="right")
            return
        if again["option"] == m["option"]:
            return
        m.update(option=None, read=again["read"], verdict="review",
                 reasons=again["reasons"] or [f"read as {m['option']} and as {again['option'] or 'unclear'}"])
    # Readings disagree, or several options are written: some were crossed out. The one left standing.
    kept = kept_option(crops_dir / c["file"])
    if kept:
        m.update(option=kept, reasons=[f"options crossed out; {kept} is the one left"],
                 verdict="right" if kept == m["key"] else "wrong")


def split(result: dict, crops_dir: Path, scheme: dict[str, float], key: dict[str, str],
          progress: Callable[[str], None] = print) -> dict:
    """Replace each question that has parts in the scheme by its parts (in place); return result.

    When the labels start again at (i) and the next question is nowhere on the sheet, its number was
    missed ("(i) 9.(क)" in the margin): the rest of the answer becomes that question, flagged."""
    with_parts = {n.split(".")[0] for n in scheme if "." in n}
    todo = [q for q in result["questions"] if q["number"] in with_parts and q["crops"] and not q.get("part_of")]
    numbers = {q["number"] for q in result["questions"]} | {q.get("part_of") for q in result["questions"]}

    def parts_of(main: str, found: list[tuple[int, str]], crops: list, flags: list[str], whole: str) -> list[dict]:
        made = []
        for idx, text in found:
            number = f"{main}.{idx}"
            part = {"number": number, "flags": list(flags), "crops": crops, "part_of": main,
                    "transcript": text, "whole_transcript": whole}
            if number not in scheme:
                part["flags"].append(f"part {number} is not in the marks scheme")
            if number in key:
                part["mcq"] = mcq_from_text(text, key[number])
                recheck_mcq(part, crops_dir)
            made.append(part)
        return made

    out = []
    done = 0
    for q in result["questions"]:
        if q not in todo:
            out.append(q)
            continue
        done += 1
        progress(f"Reading answers in parts: question {q['number']} ({done} of {len(todo)})")
        transcript = q.get("whole_transcript") or q.get("transcript")
        if transcript is None:
            texts = []
            for c in q["crops"]:
                text = (read_answer(str(crops_dir / c["file"]), q["number"])["text"] or "").strip()
                if text:
                    texts.append(text)
            transcript = NL.join(texts)
        # A number already answered whose own text starts with another, unanswered one ("34" in the margin,
        # "37. (b)" in the answer, after Q34's parts): the margin was misread; the text's number is taken.
        lead = LEAD_NUMBER.match(transcript)
        mains = {n.split(".")[0] for n in scheme}
        if (lead and lead.group(1) != q["number"] and lead.group(1) in mains and lead.group(1) not in numbers
                and any(x is not q and q["number"] in (x["number"].split(".")[0], x.get("part_of"))
                        for x in result["questions"] + out)):
            q["flags"] = [f for f in q.get("flags", []) if "comes after" not in f and "already answered" not in f]
            q["flags"].append(f"read as Q{q['number']} in the margin, but the answer starts \"{lead.group(0).strip()}\": "
                              f"taken as Q{lead.group(1)}, check it")
            q["number"] = lead.group(1)
            numbers.add(q["number"])
            if q["number"] not in with_parts:
                q["transcript"] = transcript
                out.append(q)
                continue
        # The last part of the question before can sit in this one's first crop ("(ङ) कर्मवाच्य" above
        # "5. (क) ..."): what comes before this question's own number goes back to that question.
        lines = transcript.splitlines()
        start = next((n for n, line in enumerate(lines) if n and OWN_START(q["number"]).match(line)), None)
        prev = str(int(q["number"]) - 1)
        if start is not None and split_text(NL.join(lines[:start])):
            carry = split_text(NL.join(lines[:start]))
            transcript = NL.join(lines[start:])
            have = {x["number"] for x in out}
            back = [p for p in parts_of(prev, carry, q["crops"], [], NL.join(lines[:start]))
                    if p["number"] not in have and prev in with_parts]
            at = max((i for i, x in enumerate(out) if x.get("part_of") == prev), default=len(out) - 1) + 1
            out[at:at] = back
        nxt = str(int(q["number"]) + 1)
        found = split_text(transcript, restart_ends=nxt in with_parts and nxt not in numbers)
        if not found:
            q["transcript"] = transcript
            q["flags"] = q.get("flags", []) + ["the parts of this answer could not be told apart: mark it as a whole"]
            out.append(q)
            continue
        following = found.pop()[1] if found[-1][0] == 0 else None
        out += parts_of(q["number"], found, q["crops"], q.get("flags", []), transcript)
        if following is not None:
            flag = f"Q{nxt} was found inside Q{q['number']}'s answer (its number was not read): check it"
            out += parts_of(nxt, split_text(following), q["crops"], [flag], following)
    result["questions"] = out
    return result

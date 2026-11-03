"""Tests for the rules that need no model: reading question numbers, grouping crops into questions,
cleaning MCQ answers, the marks scheme and suggested marks. Each case is a situation seen on real
sheets (SST and maths booklets, 2026-11-03).

    python -m unittest eval/test_logic.py      (from the project folder, or test_logic.py from eval/)
"""
import unittest

from app import needs_look, out_of, parse_scheme
from booklet import Crop, assign, merge_pairs, parse, same_number
from grade import suggest
from mcq import only_number, strip_number
from points import evidence_found
from qpaper import check, scheme_text
from segment import Block


def crop(label: str, conf: float = 0.9, kind: str = "", page: int = 1) -> Crop:
    return Crop(page, "x.jpg", label, conf, kind)


def numbers(crops: list[Crop]) -> list[str]:
    return [q.number for q in assign(crops)]


class ParseNumbers(unittest.TestCase):
    def test_labels_as_written(self):
        cases = {"13.": "13", "Q14.": "14", "Ans 3.": "3", "Aus 16.": "16", "34.1": "34.1", "25 b)": "25",
                 "016.": "16", "030.": "30",          # a handwritten Q read as 0
                 "(2)": "2", "②": "2", "-": None, "Faults": None, "?": None}
        for label, want in cases.items():
            self.assertEqual(parse(label)[0], want, label)

    def test_brackets_mark_a_point(self):
        self.assertTrue(parse("(2)")[1])
        self.assertTrue(parse("③")[1])           # circled 3
        self.assertFalse(parse("2.")[1])

    def test_same_number_ignores_a_stray_dot(self):
        self.assertTrue(same_number("31", "3.1"))   # "Ans 31." read as 3.1
        self.assertFalse(same_number("34.1", "34.2"))


class Grouping(unittest.TestCase):
    def test_numbered_points_stay_in_their_answer(self):
        qs = assign([crop("25"), crop("2."), crop("3."), crop("26"), crop("1."), crop("2."), crop("27")])
        self.assertEqual([q.number for q in qs], ["25", "26", "27"])
        self.assertEqual([c.kind for c in qs[0].crops], ["question", "point", "point"])

    def test_mcqs_one_after_another_are_questions(self):
        self.assertEqual(numbers([crop(str(n)) for n in range(1, 9)]), [str(n) for n in range(1, 9)])

    def test_continuation_joins_the_answer_before(self):
        qs = assign([crop("12"), crop("", None, "continuation", page=2), crop("13", page=2)])
        self.assertEqual([len(q.crops) for q in qs], [2, 1])

    def test_unsure_next_number_is_accepted(self):
        self.assertEqual(numbers([crop("3"), crop("4", 0.3)]), ["3", "4"])

    def test_unsure_unlikely_number_stays_with_its_answer(self):
        qs = assign([crop("22"), crop("2", 0.3), crop("40", 0.3)])   # => arrows in maths working
        self.assertEqual([q.number for q in qs], ["22"])
        self.assertEqual(qs[0].flags, [])                           # not worth the teacher's time

    def test_unsure_mark_that_could_be_the_next_question_is_flagged(self):
        qs = assign([crop("22"), crop("25", 0.3)])
        self.assertEqual([q.number for q in qs], ["22"])
        self.assertTrue(qs[0].flags)

    def test_first_number_is_taken_but_flagged_when_unsure(self):
        qs = assign([crop("", None, "continuation"), crop("25 b)", 0.4)])
        self.assertEqual(qs[-1].number, "25")
        self.assertTrue(any("not for sure" in f for f in qs[-1].flags))

    def test_repeated_number_is_flagged(self):
        qs = assign([crop("13"), crop("14"), crop("15"), crop("13")])
        self.assertIn("already answered", qs[-1].flags[0])

    def test_rewritten_question_joins_the_first(self):
        qs = assign([crop("5"), crop("6"), crop("", None, "continuation", page=2), crop("6", page=2), crop("7", page=2)])
        self.assertEqual([q.number for q in qs], ["5", "6", "7"])
        self.assertEqual(len(qs[1].crops), 3)

    def test_dot_misread_is_corrected_when_it_is_the_next_question(self):
        self.assertEqual(numbers([crop("34"), crop("3.5")]), ["34", "35"])

    def test_unsure_number_before_the_same_clear_number_becomes_unknown(self):
        qs = assign([crop("", None, "continuation"), crop("2", 0.4), crop("2"), crop("3")])   # "1." read as 2
        self.assertEqual([q.number for q in qs if q.number != "start"], ["?", "2", "3"])

    def test_big_jump_is_flagged_even_when_sure(self):
        qs = assign([crop("23"), crop("24"), crop("34")])                                 # 37 read as 34
        self.assertIn("comes after Q24", qs[-1].flags[0])
        qs = assign([crop("25"), crop("21")])                                             # 26 read as 21
        self.assertIn("comes after Q25", qs[-1].flags[0])
        self.assertEqual(assign([crop("13"), crop("14"), crop("15")])[-1].flags, [])

    def test_q_read_as_a_digit(self):                  # "Q.25)" read as 8.25, "Q.1)" as 8.1
        self.assertEqual(numbers([crop("8.24"), crop("9.25"), crop("0.26")]), ["24", "25", "26"])
        self.assertEqual(numbers([crop("", None, "continuation"), crop("8.1"), crop("8.2"), crop("9.3")]),
                         ["start", "1", "2", "3"])
        self.assertEqual(numbers([crop("34"), crop("34.1"), crop("34.2")]), ["34", "34.1", "34.2"])

    def test_sign_in_margin_is_a_note(self):
        qs = assign([crop("22"), crop("-")])
        self.assertEqual(numbers([crop("22"), crop("-")]), ["22"])
        self.assertEqual(qs[0].crops[1].kind, "note")


class MergeLines(unittest.TestCase):
    def block(self, y):
        return Block(y, y + 40, (10, y, 60, y + 20))

    def test_q_and_ans_lines_become_one_crop(self):
        blocks, reads = merge_pairs([self.block(100), self.block(150)], [("Q1.", 0.3), ("1", 0.9)], 1200)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(reads[0], ("1", 0.9))      # the surer reading is kept
        self.assertEqual((blocks[0].top, blocks[0].bottom), (100, 190))

    def test_different_questions_stay_apart(self):
        blocks, _ = merge_pairs([self.block(100), self.block(150)], [("14", 0.9), ("15", 0.9)], 1200)
        self.assertEqual(len(blocks), 2)

    def test_unreadable_number_on_the_next_line_is_its_own_question(self):
        blocks, _ = merge_pairs([self.block(100), self.block(150)], [("8.3", 0.9), ("?.4)", 0.3)], 1200)
        self.assertEqual(len(blocks), 2)

    def test_q_read_as_digit_keeps_two_digit_numbers_apart(self):
        blocks, _ = merge_pairs([self.block(100), self.block(150)], [("8.25", 0.9), ("8.26", 0.9)], 1200)
        self.assertEqual(len(blocks), 2)

    def test_far_apart_lines_stay_apart(self):
        blocks, _ = merge_pairs([self.block(100), self.block(400)], [("6", 0.9), ("6", 0.9)], 1200)
        self.assertEqual(len(blocks), 2)


class McqText(unittest.TestCase):
    def test_question_number_is_removed(self):
        self.assertEqual(strip_number("Q1. Ans 1. (C) 1:4"), "(C) 1:4")
        self.assertEqual(strip_number("02.\nAns 2. (D) Sphere"), "(D) Sphere")
        self.assertEqual(strip_number("13. B) Garhwali").strip(), "B) Garhwali")

    def test_answer_text_is_kept(self):
        self.assertEqual(strip_number("Both (A) and (R) are true"), "Both (A) and (R) are true")
        self.assertEqual(strip_number("2x + 3 = 5"), "2x + 3 = 5")

    def test_only_a_number(self):
        self.assertTrue(only_number("015"))
        self.assertTrue(only_number("Q17."))
        self.assertFalse(only_number("(B) 4000"))


class Marks(unittest.TestCase):
    def test_scheme(self):
        s = parse_scheme("1-3:1, 4:2, 34.1:1.5")
        self.assertEqual(s, {"1": 1, "2": 1, "3": 1, "4": 2, "34.1": 1.5})
        self.assertEqual(out_of(s, "4"), 2)
        self.assertEqual(out_of(parse_scheme("34:4"), "34.2"), 4)

    def test_scheme_text_and_check(self):
        secs = [{"name": "A", "first": 1, "last": 20, "marks": 1, "type": "mcq"},
                {"name": "B", "first": 21, "last": 24, "marks": 2, "type": "short"}]
        self.assertEqual(scheme_text(secs), "1-20:1, 21-24:2")
        self.assertEqual(check(secs, 28), [])
        self.assertTrue(check(secs, 80))                       # doesn't add up
        self.assertTrue(check(secs[1:], 8))                    # questions 1-20 missing

    def test_suggested_marks(self):
        self.assertEqual(suggest(3, 3, 3), 3)
        self.assertEqual(suggest(4, 3, 3), 3)                  # more points than needed
        self.assertEqual(suggest(1, 3, 5), 1.5)                # half marks
        self.assertEqual(suggest(0, 4, 5), 0)

    def test_evidence_must_be_in_the_answer(self):
        answer = "Gandhi ji later on designed his swaraj flag with a charkha in the middle."
        self.assertTrue(evidence_found("designed his swaraj flag with a charkha", answer))
        self.assertFalse(evidence_found("Raja Ravi Varma painted gods", answer))

    def test_what_the_teacher_must_look_at(self):
        self.assertFalse(needs_look({"flags": [], "mcq": {"verdict": "right"}}))
        self.assertTrue(needs_look({"flags": [], "mcq": {"verdict": "wrong"}}))
        self.assertTrue(needs_look({"flags": ["x"]}))
        full = {"empty": False, "suggested": 3, "out_of": 3}
        self.assertFalse(needs_look({"flags": [], "grading": full}))
        self.assertTrue(needs_look({"flags": [], "grading": {**full, "suggested": 2}}))
        self.assertTrue(needs_look({"flags": [], "grading": {**full, "empty": True}}))


if __name__ == "__main__":
    unittest.main()

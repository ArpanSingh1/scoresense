"""Tests for the rules that need no model: reading question numbers, grouping crops into questions,
cleaning MCQ answers, the marks scheme and suggested marks. Each case is a situation seen on real
sheets (SST and maths booklets, 2026-11-03).

    python -m unittest eval/test_logic.py      (from the project folder, or test_logic.py from eval/)
"""
import unittest

from app import capped_total, needs_look, out_of, parse_scheme, scheme_total
from booklet import Crop, assign, fit_to_paper, merge_pairs, parse, same_number
from grade import close_match, other_points, split_by_text, suggest
from mcq import label_confidence, only_number, strip_number
from parts import mcq_from_text, split, split_text
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

    def test_skipped_number_read_as_the_next_is_taken_back(self):   # Science booklet: "Q22)" read as 8.23
        qs = assign([crop("8.21", 0.64), crop("8.23", 0.63), crop("", None, "continuation", page=2),
                     crop("9.23", 0.61, page=3), crop("8.24", 0.79, page=4)])
        self.assertEqual([q.number for q in qs], ["21", "22", "23", "24"])
        self.assertTrue(any("taken as Q22" in f for f in qs[1].flags))

    def test_rewritten_number_after_a_real_gap_is_kept(self):       # Q22 was answered: 23 really is 23
        self.assertEqual(numbers([crop("21"), crop("22"), crop("23"), crop("", None, "continuation", page=2),
                                  crop("23", page=2), crop("24", page=2)]), ["21", "22", "23", "24"])

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

    def test_number_not_in_paper_is_fitted(self):         # SST booklet: Q31 read as 71
        qs = [{"number": n, "flags": []} for n in ("30", "71", "33", "34.1")]
        fit_to_paper(qs, set(range(1, 38)))
        self.assertEqual([q["number"] for q in qs], ["30", "31", "33", "34.1"])
        self.assertIn("taken as Q31", qs[1]["flags"][0])
        qs = [{"number": n, "flags": ["Q88 comes after Q30: check"]} for n in ("30", "88", "33")]
        fit_to_paper(qs, set(range(1, 38)))
        self.assertEqual(qs[1]["number"], "88")
        self.assertEqual(qs[1]["flags"], ["Q88 is not in the paper: check the number"])

    def test_number_in_a_one_question_gap_is_fitted(self):   # SST booklet: 25, "21" (26), 27 ... 21 again
        qs = [{"number": n, "flags": [], "crops": [{"page": p}]} for n, p in
              (("25", 4), ("21", 6), ("27", 7), ("29", 9), ("36.3", 20), ("21", 21), ("22", 21))]
        qs[5]["flags"] = ["Q21 was already answered on page 6"]
        fit_to_paper(qs, set(range(1, 38)))
        self.assertEqual([q["number"] for q in qs], ["25", "26", "27", "29", "36.3", "21", "22"])
        self.assertIn("taken as Q26", qs[1]["flags"][0])
        self.assertEqual(qs[5]["flags"], [])
        qs = [{"number": n, "flags": []} for n in ("25", "13", "27")]      # 13 is not one digit off 26
        fit_to_paper(qs, set(range(1, 38)))
        self.assertEqual(qs[1]["number"], "13")

    def test_skipped_section_flags_the_jump_only(self):  # SST booklet: 18, 25-36, then 21-23 at the end
        qs = assign([crop("17"), crop("18"), crop("25"), crop("26"), crop("27"), crop("29"), crop("36"),
                     crop("21"), crop("22")])
        self.assertEqual([q.number for q in qs], ["17", "18", "25", "26", "27", "29", "36", "21", "22"])
        self.assertEqual([bool(q.flags) for q in qs], [False, False, True, False, False, False, True, True, False])

    def test_q_read_as_a_digit(self):                  # "Q.25)" read as 8.25, "Q.1)" as 8.1
        self.assertEqual(numbers([crop("8.24"), crop("9.25"), crop("0.26")]), ["24", "25", "26"])
        self.assertEqual(numbers([crop("", None, "continuation"), crop("8.1"), crop("8.2"), crop("9.3")]),
                         ["start", "1", "2", "3"])
        self.assertEqual(numbers([crop("34"), crop("34.1"), crop("34.2")]), ["34", "34.1", "34.2"])

    def test_q_read_as_a_digit_without_the_dot(self):  # "Q.29)" read as 829
        self.assertEqual(numbers([crop("8.28"), crop("829", 0.3), crop("8.30")]), ["28", "29", "30"])

    def test_old_number_inside_a_later_answer_is_a_sub_part(self):   # "(b)" read as 6, "ii)" as 11
        qs = assign([crop(str(n)) for n in range(1, 27)] + [crop("6"), crop("27"), crop("11"), crop("28")])
        self.assertEqual([q.number for q in qs][-3:], ["26", "27", "28"])
        self.assertFalse(any(q.flags for q in qs))

    def test_a_year_is_not_a_question(self):
        self.assertIsNone(parse("2025")[0])

    def test_unreadable_number_between_two_takes_the_gap(self):
        qs = assign([crop("8.1"), crop("?.", 0.3), crop("8.3")])         # Q2 read as "?."
        self.assertEqual([q.number for q in qs], ["1", "2", "3"])
        self.assertIn("from the order", qs[1].flags[0])

    def test_question_mark_without_a_gap_stays_a_note(self):
        qs = assign([crop("22"), crop("?", 0.3), crop("23")])
        self.assertEqual([q.number for q in qs], ["22", "23"])

    def test_sign_in_margin_is_a_note(self):
        qs = assign([crop("22"), crop("-")])
        self.assertEqual(numbers([crop("22"), crop("-")]), ["22"])
        self.assertEqual(qs[0].crops[1].kind, "note")


class TextNumbers(unittest.TestCase):
    def test_next_question_inside_an_answer_is_split_off(self):   # SST: 28 written on under 27
        qs = [{"number": "27", "flags": [], "crops": ["a", "b", "c"]}]
        texts = ["27. Challenges are\n1. Lack of internal democracy", "3. Money and muscle power\n28. Three features are",
                 "2. Jurisdictions are in the Constitution"]
        split_by_text(qs, qs[0], texts, {"27": {}, "28": {}})
        self.assertEqual([q["number"] for q in qs], ["27", "28"])
        self.assertEqual(qs[0]["crops"], ["a", "b"])
        self.assertEqual(qs[1]["crops"], ["b", "c"])
        self.assertTrue(qs[1]["transcript"].startswith("28. Three"))
        self.assertTrue(qs[0]["transcript"].endswith("muscle power"))

    def test_points_numbered_like_the_next_question_stay(self):
        qs = [{"number": "2", "flags": [], "crops": ["a"]}]
        split_by_text(qs, qs[0], ["1. one\n3. three"], {"2": {}, "3": {}})          # small numbers: points
        self.assertEqual(len(qs), 1)
        qs = [{"number": "27", "flags": [], "crops": ["a"]}, {"number": "28", "flags": [], "crops": ["b"]}]
        split_by_text(qs, qs[0], ["27. x\n28. y"], {"27": {}, "28": {}})            # 28 is answered already
        self.assertEqual(len(qs), 2)

    def test_repeated_number_takes_the_number_its_answer_starts_with(self):   # "34" in the margin, "37. (b)"
        result = {"questions": [{"number": "34.1", "flags": [], "crops": []},
                                {"number": "34", "flags": ["Q34 comes after Q24: check"], "crops": ["c"],
                                 "transcript": "37. (b)"}]}
        split(result, None, {"34.1": 1, "34.2": 1, "37": 5}, {}, progress=lambda m: None)
        self.assertEqual([q["number"] for q in result["questions"]], ["34.1", "37"])
        self.assertIn("taken as Q37", result["questions"][1]["flags"][0])

    def test_other_relevant_points_need_their_own_quote(self):
        answer = "Agenda 21 was adopted by all. Leaders signed the Declaration on Climate Change."
        found = [{"point": "Agenda 21", "quote": "Agenda 21 was adopted"},
                 {"point": "made up", "quote": "Paris agreement was signed"},
                 {"point": "Agenda again", "quote": "Agenda 21 was adopted by all"},
                 {"point": "Climate", "quote": "signed the Declaration on Climate Change"}]
        got = other_points(found, answer, [], 2)
        self.assertEqual([g["quote"] for g in got], ["Agenda 21 was adopted", "signed the Declaration on Climate Change"])
        self.assertTrue(all(g["other"] and g["made"] for g in got))
        self.assertEqual(other_points(found, answer, ["Agenda 21 was adopted by all."], 1)[0]["point"],
                         "Other relevant point: Climate")


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

    def test_bare_question_mark_is_not_joined(self):
        blocks, _ = merge_pairs([self.block(100), self.block(150)], [("8.1", 0.9), ("?.", 0.3)], 1200)
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


class MarkingScheme(unittest.TestCase):
    TEXT = """General Instructions:
1. You are aware that evaluation is the most important process
2. Evaluation policy is a confidential policy
Section A
19 (A) Chancellor Duke Metternich 10 H 1
20 (C) Buddhism 54 H 1
Section B
21 Why was the Indian subcontinent central to the flow of the trading network? Explain.
(i) The silk routes are a good example of vibrant pre-modern trade and cultural links. 54 H 2X1=2
(ii) It connected Europe and Africa in the west and China in the east.
(iii) Any other relevant point.
Any two points to be explained.
22 (a) "Minerals occur in igneous rocks." Explain.
(i) Minerals may occur in the cracks, crevices, faults or joints.
(ii) Any other relevant point.
Any two points to be explained. 43 G 2X1=2
OR
(b) "The ocean water contains minerals." Explain.
(i) Common salt, magnesium and bromine are largely derived from ocean water.
23.1 Which language? 1
(i) English"""

    def test_key_points_and_marks(self):
        from unittest import mock
        from rubric import marks_scheme, scheme_from_text
        with mock.patch("rubric.chat") as chat:
            r = scheme_from_text(self.TEXT, None, progress=lambda m: None)
            chat.assert_not_called()                           # everything here has (i), (ii) items
        self.assertEqual(r["key"], {"19": "A", "20": "C"})   # the numbered instructions are not questions
        self.assertEqual(r["points"]["21"], {"points": [
            "The silk routes are a good example of vibrant pre-modern trade and cultural links.",
            "It connected Europe and Africa in the west and China in the east."], "needed": 2, "other": True,
            "question": "Why was the Indian subcontinent central to the flow of the trading network? Explain."})
        self.assertEqual(len(r["points"]["22"]["points"]), 2)   # both alternatives of the OR
        self.assertEqual(r["points"]["23.1"]["points"], ["English"])
        self.assertEqual(marks_scheme({**r["marks"], "23.1": 1}), "19-20:1, 21-22:2, 23.1:1")


class McqText(unittest.TestCase):
    def test_question_number_is_removed(self):
        self.assertEqual(strip_number("Q1. Ans 1. (C) 1:4"), "(C) 1:4")
        self.assertEqual(strip_number("02.\nAns 2. (D) Sphere"), "(D) Sphere")
        self.assertEqual(strip_number("13. B) Garhwali").strip(), "B) Garhwali")
        self.assertEqual(strip_number("Q.11) CD) Auxins"), "(D) Auxins")

    def test_answer_text_is_kept(self):
        self.assertEqual(strip_number("Both (A) and (R) are true"), "Both (A) and (R) are true")
        self.assertEqual(strip_number("2x + 3 = 5"), "2x + 3 = 5")

    def test_letter_is_weighed_against_other_letters(self):
        import math
        tokens, alts = [("(D", math.log(0.2))], [[("(D", math.log(0.2)), ("Q", math.log(0.15)), ("9", math.log(0.13))]]
        self.assertEqual(label_confidence(tokens, "D", alts), 1.0)          # unsure whether to copy "Q.7)"
        alts = [[("(D", math.log(0.2)), ("(B", math.log(0.2))]]
        self.assertEqual(label_confidence(tokens, "D", alts), 0.5)          # D or B: unclear

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


class HindiParts(unittest.TestCase):
    def test_parts_split_at_their_labels(self):
        text = "4. (क) दादाजी अब चल नहीं सकते,\n(ख) कैप्टन द्वारा\n(वा) उससे चलकर आया गया,\n(ध) कर्तृवाच्य\n(इ.) कर्मवाच्य"
        self.assertEqual([i for i, _ in split_text(text)], [1, 2, 3, 4, 5])   # (वा) is (ग), (ध) (घ), (इ.) (ङ)
        self.assertEqual(split_text(text)[3][1], "कर्तृवाच्य")

    def test_both_labels_and_the_number_between_them(self):
        parts = split_text("(i) 9.(क) (A)\n(ii)(ख) (D)\n(iii) (D)")
        self.assertEqual(parts, [(1, "(A)"), (2, "(D)"), (3, "(D)")])

    def test_restart_at_the_first_label_is_the_next_question(self):
        parts = split_text("8 (क) a\n(ख) b\n(ग) c\n(i) (A)\n(ii) (D)", restart_ends=True)
        self.assertEqual(parts[-1][0], 0)
        self.assertEqual(split_text("(क) a\n(ख) b\n(i) c"), [(1, "a"), (2, "b\n(i) c")])   # else it's text

    def test_mcq_part_with_crossed_out_options_goes_to_the_teacher(self):
        self.assertEqual(mcq_from_text("(D) I और II", "D")["verdict"], "right")
        self.assertEqual(mcq_from_text("(A) (C) (D)", "D")["verdict"], "review")

    def test_any_four_of_five(self):
        s = parse_scheme("1.1-1.3:1, 1.4-1.5:2, 3.1-3.5:1, 3:4, 12:6")
        self.assertEqual(s["1.5"], 2)
        self.assertEqual(scheme_total(s), 7 + 4 + 6)
        self.assertEqual(capped_total([("3.1", 1), ("3.2", 1), ("3.3", 1), ("3.4", 1), ("3.5", 1), ("12", 5)], s), 9)

    def test_hindi_quotes_are_found(self):
        self.assertTrue(evidence_found("सरल वाक्य", "सरल वाक्य"))
        self.assertTrue(evidence_found("कर्तृवाच्य", "कर्तृवाच्य"))        # a one-word answer
        self.assertFalse(evidence_found("the", "the answer is long enough here"))


class ShortAnswers(unittest.TestCase):                  # Hindi booklet: right answers with misread letters
    def test_misread_right_answers_match(self):
        for point, said in [
                ("कर्मवाच्य: कैप्टन द्वारा नेताजी की मूर्ति पर असली चश्मा लगाया जाता था",
                 "कैप्टन द्वारा नेलाजी की मूर्ति पर असली चर्म लगाया जाता था,"),
                ("भाववाच्य: उससे चलकर आया गया था", "उससे चलकर आया गाया,"),
                ("कर्तृवाच्य", "कर्तव्य"), ("कर्मवाच्य", "कर्मवाद्य"),
                ("भेद: संज्ञा आश्रित उपवाक्य", "कि वे भी आभकथा लिखें। संजा आश्रित उपवाक्य")]:
            self.assertTrue(close_match(point, said), said)

    def test_wrong_answers_do_not(self):
        for point, said in [
                ("कर्तृवाच्य", "कर्मवाच्य"), ("कर्मवाच्य", "भाववाच्य"), ("उपमा अलंकार", "रूपक अलंकार"),
                ("भेद: संज्ञा आश्रित उपवाक्य", "कि वे भी आत्मकथा लिखें। विशेषण आश्रित उपवाक्य"),
                ("भाववाच्य: उससे चलकर आया गया था", "वह चलकर आया था"),
                ("कर्मवाच्य: कैप्टन द्वारा नेताजी की मूर्ति पर असली चश्मा लगाया जाता था",
                 "कैप्टन नेताजी की मूर्ति पर असली चश्मा लगाता था"),
                ("क्रियाविशेषण", "क्रिया"), ("Rio de Janeiro Earth Summit, 1992", "Paris summit 2015")]:
            self.assertFalse(close_match(point, said), said)
        self.assertFalse(close_match("विशेषण (गुणवाचक)", "क्रियाविशेषण"))  # another part of speech


if __name__ == "__main__":
    unittest.main()

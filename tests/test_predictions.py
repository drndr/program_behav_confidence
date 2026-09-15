from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from predictions import (RE2_INPUT_NOTE, confidence_prompt, parse_prediction, parse_verdict,
                         prediction_prompt, same_value, score_prediction, template_sha256)

CRUX = {"dataset": "cruxeval", "entrypoint": "f", "code": "def f(a):\n    return a", "input": "'x'", "output": "'x'"}
RE2 = {"dataset": "re2", "entrypoint": "pair", "code": "def pair(a):\n    return (a, a)",
       "input": '{"self": {}, "args": {"a": 1}, "kwargs": {}}', "output": "[1, 1]"}
STDIN = {"dataset": "re2", "entrypoint": "", "code": "print(input())", "input": '{"input": "hi"}', "output": '"hi"'}


class PromptTests(unittest.TestCase):
    def test_entrypoint_and_stdin_wording(self):
        self.assertIn("returned by `pair`", prediction_prompt(RE2))
        stdin = prediction_prompt(STDIN)
        self.assertIn("prints", stdin)
        self.assertNotIn("returned by", stdin)

    def test_input_note_only_for_re2(self):
        self.assertNotIn(RE2_INPUT_NOTE, prediction_prompt(CRUX))
        self.assertIn(RE2_INPUT_NOTE, prediction_prompt(RE2))
        self.assertIn(RE2_INPUT_NOTE, prediction_prompt(STDIN))

    def test_reference_never_included(self):
        example = dict(CRUX, output="SECRET_REFERENCE")
        self.assertNotIn("SECRET_REFERENCE", prediction_prompt(example))
        self.assertNotIn("SECRET_REFERENCE", confidence_prompt(example, "'x'"))

    def test_confidence_prompt_shows_candidate_and_asks_binary(self):
        prompt = confidence_prompt(RE2, "(1, 1)")
        self.assertIn("[CANDIDATE]\n(1, 1)\n[/CANDIDATE]", prompt)
        self.assertIn("<verdict>", prompt)
        self.assertNotIn("<prediction>", prompt)

    def test_template_hash_is_stable(self):
        self.assertEqual(template_sha256(), template_sha256())
        self.assertEqual(len(template_sha256()), 64)


class ResponseTests(unittest.TestCase):
    def test_prediction_requires_exactly_one_tag(self):
        self.assertEqual(parse_prediction("text <prediction> [1, 2] </prediction> more"), "[1, 2]")
        for text in ("[1, 2]", "<prediction>1</prediction><prediction>2</prediction>", "<prediction>1"):
            with self.assertRaises(ValueError):
                parse_prediction(text)

    def test_verdict_accepts_only_two_words(self):
        self.assertEqual(parse_verdict("<verdict>Correct</verdict>"), "correct")
        self.assertEqual(parse_verdict("<verdict> incorrect </verdict>"), "incorrect")
        for text in ("<verdict>0.9</verdict>", "<verdict>mostly correct</verdict>", "correct"):
            with self.assertRaises(ValueError):
                parse_verdict(text)


class ScoringTests(unittest.TestCase):
    def test_preserve_nested_types(self):
        self.assertFalse(same_value([True], [1]))
        self.assertFalse(same_value((1,), [1]))
        self.assertFalse(same_value({True: 2}, {1: 2}))
        self.assertTrue(same_value({"a": [1, "x"]}, {"a": [1, "x"]}))

    def test_cruxeval_is_type_strict(self):
        self.assertTrue(score_prediction(CRUX, "<prediction>'x'</prediction>")["correct"])
        self.assertFalse(score_prediction(CRUX, "<prediction>\"y\"</prediction>")["correct"])
        example = dict(CRUX, output="(1, 2)")
        self.assertFalse(score_prediction(example, "<prediction>[1, 2]</prediction>")["correct"])

    def test_re2_prediction_is_normalized_like_upstream_references(self):
        self.assertTrue(score_prediction(RE2, "<prediction>(1, 1)</prediction>")["correct"])
        keyed = dict(RE2, output='{"1": "x"}')
        self.assertTrue(score_prediction(keyed, "<prediction>{1: 'x'}</prediction>")["correct"])
        self.assertFalse(score_prediction(RE2, "<prediction>(1, 2)</prediction>")["correct"])

    def test_statuses_for_bad_responses(self):
        self.assertEqual(score_prediction(CRUX, "no tags")["status"], "invalid_response")
        self.assertEqual(score_prediction(CRUX, "<prediction>f(1)</prediction>")["status"], "invalid_prediction")
        self.assertEqual(score_prediction(RE2, "<prediction>{1, 2}</prediction>")["status"], "invalid_prediction")

    def test_serialized_objects_are_flagged_not_scored(self):
        example = dict(RE2, output='"array([1])"')
        result = score_prediction(example, "<prediction>'array([1])'</prediction>")
        self.assertEqual((result["status"], result["correct"], result["exact_match"]), ("needs_scoring_policy", None, True))
        plain = score_prediction(dict(RE2, output='"4"'), "<prediction>4</prediction>")
        self.assertEqual((plain["status"], plain["correct"]), ("scored", False))


if __name__ == "__main__":
    unittest.main()

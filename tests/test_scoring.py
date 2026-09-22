import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scoring import SCORING_MODES, parse_prediction, prediction_logprobs, same_value, score_prediction
import prompt_template
from prompt_template import (RE2_INPUT_NOTE, prediction_prompt, prediction_template_sha256, task_sentence,
                             verification_prompt, verification_template_sha256, verification_wording_final)

CRUX = {"dataset": "cruxeval", "entrypoint": "f", "code": "def f(a):\n    return a", "input": "'x'", "output": "'x'"}
RE2 = {"dataset": "re2", "entrypoint": "pair", "code": "def pair(a):\n    return (a, a)",
       "input": '{"self": {}, "args": {"a": 1}, "kwargs": {}}', "output": "[1, 1]"}
STDIN = {"dataset": "re2", "entrypoint": "", "code": "print(input())", "input": '{"input": "hi"}', "output": '"hi"'}


class PromptTests(unittest.TestCase):
    def test_dataset_specific_prompt_context(self):
        self.assertIn("returned by `pair`", prediction_prompt(RE2))
        stdin = prediction_prompt(STDIN)
        self.assertIn("prints", stdin)
        self.assertNotIn("returned by", stdin)
        self.assertNotIn(RE2_INPUT_NOTE, prediction_prompt(CRUX))
        self.assertIn(RE2_INPUT_NOTE, prediction_prompt(RE2))
        self.assertIn(RE2_INPUT_NOTE, prediction_prompt(STDIN))

    def test_reference_never_included(self):
        example = dict(CRUX, output="SECRET_REFERENCE")
        self.assertNotIn("SECRET_REFERENCE", prediction_prompt(example))
        self.assertNotIn("SECRET_REFERENCE", verification_prompt(example, "'x'"))

    def test_verification_prompt_structure(self):
        # Wording is not decided; the structure is: what the predictor saw, plus the candidate.
        prompt = verification_prompt(RE2, "(1, 1)")
        self.assertIn(task_sentence(RE2), prompt)
        self.assertIn("[CODE]\n" + RE2["code"] + "\n[/CODE]", prompt)
        self.assertIn(RE2_INPUT_NOTE, prompt)
        self.assertIn("[CANDIDATE]\n(1, 1)\n[/CANDIDATE]", prompt)
        self.assertNotIn("<prediction>", prompt)
        self.assertNotIn("\n\n\n", prompt)  # empty wording slots leave no blank paragraphs
        self.assertFalse(verification_wording_final())
        with patch.object(prompt_template, "VERIFICATION_HEADER", "Judge."), \
                patch.object(prompt_template, "VERIFICATION_ANSWER", "Answer in <verdict> tags."):
            self.assertTrue(verification_wording_final())
            self.assertTrue(verification_prompt(RE2, "(1, 1)").startswith("Judge.\n\n"))

    def test_template_hashes_are_independent(self):
        before = prediction_template_sha256(), verification_template_sha256()
        self.assertEqual(before, (prediction_template_sha256(), verification_template_sha256()))
        with patch.object(prompt_template, "VERIFICATION_HEADER", "Judge."):
            self.assertEqual(prediction_template_sha256(), before[0])
            self.assertNotEqual(verification_template_sha256(), before[1])


class ResponseTests(unittest.TestCase):
    def test_prediction_requires_exactly_one_tag(self):
        self.assertEqual(parse_prediction("text <prediction> [1, 2] </prediction> more"), "[1, 2]")
        for text in ("[1, 2]", "<prediction>1</prediction><prediction>2</prediction>", "<prediction>1"):
            with self.assertRaises(ValueError):
                parse_prediction(text)


def tokens(*pairs):
    return {"content": [{"token": t, "bytes": list(t.encode("utf-8")), "logprob": lp} for t, lp in pairs]}


class LogprobTests(unittest.TestCase):
    ANSWER = [("<p", -0.1), ("red", -0.1), ("iction", -0.1), (">'", -0.5), ("4", -1.0), ("'", -0.2),
              ("</", -0.1), ("prediction", -0.1), (">", -0.1)]
    TEXT = "<prediction>'4'</prediction>"

    def test_mean_over_content_tokens_including_tag_adjacent_ones(self):
        result = prediction_logprobs(self.TEXT, tokens(*self.ANSWER))
        self.assertEqual(result["token_count"], 3)  # ">'", "4", "'"
        self.assertAlmostEqual(result["mean_logprob"], (-0.5 - 1.0 - 0.2) / 3)
        self.assertAlmostEqual(result["geometric_mean_probability"], math.exp(result["mean_logprob"]))

    def test_tolerates_provider_stream_variants(self):
        # A </think> token before, an end marker after, or a dropped first tag token.
        variants = ([("</think>", -0.01)] + self.ANSWER + [("<|end|>", -0.01)],
                    self.ANSWER[1:], self.ANSWER[:6] + self.ANSWER[7:])
        for content in variants:
            result = prediction_logprobs(self.TEXT, tokens(*content))
            self.assertEqual(result["token_count"], 3, content)

    def test_refuses_streams_that_do_not_reproduce_the_answer(self):
        garbled = [("<p", -0.1), ("red", -0.1), ("iction", -0.1), (">'", -0.5), ("5", -1.0), ("'", -0.2), ("</", -0.1)]
        self.assertIsNone(prediction_logprobs(self.TEXT, tokens(*garbled)))
        self.assertIsNone(prediction_logprobs(self.TEXT, tokens(("<p", -0.1))))
        self.assertIsNone(prediction_logprobs(self.TEXT, {"content": []}))
        self.assertIsNone(prediction_logprobs(self.TEXT, None))
        self.assertIsNone(prediction_logprobs("no tags", tokens(*self.ANSWER)))
        missing = {"content": [{"token": t, "logprob": None} for t, _ in self.ANSWER]}
        self.assertIsNone(prediction_logprobs(self.TEXT, missing))

    def test_whitespace_padding_and_multibyte_content(self):
        text = "<prediction>\n  [1, 'é']\n</prediction>"
        content = [("<prediction>", -0.1), ("\n  ", -0.3), ("[1", -0.2), (", '", -0.2), ("é", -0.4), ("']", -0.2),
                   ("\n", -0.3), ("</prediction>", -0.1)]
        result = prediction_logprobs(text, tokens(*content))
        self.assertEqual(result["token_count"], 4)  # padding tokens excluded
        self.assertAlmostEqual(result["mean_logprob"], (-0.2 - 0.2 - 0.4 - 0.2) / 4)


class ScoringTests(unittest.TestCase):
    def normalized(self, reference, prediction):
        return score_prediction(dict(RE2, output=json.dumps(reference)),
                                f"<prediction>{prediction}</prediction>", "normalized")

    def test_every_mode_labels_every_prediction(self):
        # A completed response is always right or wrong; only the status says why.
        for mode in SCORING_MODES:
            for example, text in ((CRUX, "<prediction>f(1)</prediction>"), (RE2, "<prediction>{1, 2}</prediction>"),
                                  (dict(RE2, output='"GaussianNB(priors=None)"'), "<prediction>OtherClass()</prediction>"),
                                  (dict(RE2, output='"array([1, ..., 4])"'), "<prediction>array([1,2,3,4])</prediction>")):
                self.assertIs(score_prediction(example, text, mode)["correct"], False, (mode, text))
            result = score_prediction(CRUX, "no tags", mode)
            self.assertEqual((result["correct"], result["status"]), (False, "invalid_response"))
        for example, text in ((CRUX, "<prediction>f(1)</prediction>"),
                              (RE2, "<prediction>{1, 2}</prediction>")):
            self.assertEqual(score_prediction(example, text)["status"], "invalid_prediction")

    def test_normalized_complex_and_symbolic_outputs(self):
        self.assertFalse(self.normalized("(-5+10j)", "(-11+2j)")["correct"])
        self.assertTrue(self.normalized("(-5+10j)", "-5 + 10j")["correct"])
        reference = ["2 - 2*cos(phi)", {}, []]
        self.assertTrue(self.normalized(reference, "(2-2*cos(phi), {}, [])")["correct"])
        self.assertFalse(self.normalized(reference, "(2-3*cos(phi), {}, [])")["correct"])

    def test_normalized_array_shape_values_and_exactness(self):
        self.assertTrue(self.normalized("array([1., 2.])", "array([1,2], dtype=float64)")["correct"])
        self.assertFalse(self.normalized("array([1., 2.])", "array([1,2,3])")["correct"])
        self.assertFalse(self.normalized("array([1., 2.])", "array([[1,2]])")["correct"])
        self.assertFalse(self.normalized("array([1.0])", "array([1.00000001])")["correct"])
        self.assertTrue(self.normalized("[[1 2]\n [3 4]]", "array([[1,2],[3,4]])")["correct"])
        self.assertTrue(self.normalized("array([1, 2])", "[1 2]")["correct"])
        self.assertTrue(self.normalized("[ True False]", "array([True,False])")["correct"])
        self.assertFalse(self.normalized("[ True False]", "array([1,0])")["correct"])
        self.assertFalse(self.normalized("array([1, ..., 4])", "array([1,2,3,4])")["correct"])

    def test_model_text_is_never_executed(self):
        call = "__import__('os').system('must not execute')"
        with patch("os.system", side_effect=AssertionError("Executed model text")) as system:
            for reference, prediction in (("array([1])", call),
                                          ("prettyForm('x',0,0)", f"prettyForm({call})"),
                                          ("GaussianNB()", f"GaussianNB(x={call})")):
                self.normalized(reference, prediction)
            system.assert_not_called()

    def test_prettyform_display_normalization(self):
        reference = "prettyForm('{Q.integer, Q.positive}',0,0)"
        for prediction in ("{'Q.positive', 'Q.integer'}", "'{Q.integer, Q.positive}'",
                           "prettyForm('{Q.integer, Q.positive}',1,2)"):
            self.assertTrue(self.normalized(reference, prediction)["correct"])
        for prediction in ("{'Q.integer'}", "{'Q.integer', 'Q.negative'}",
                           "{'Q.integer', 'Q.positive', 'Q.real'}"):
            self.assertFalse(self.normalized(reference, prediction)["correct"])
        self.assertTrue(self.normalized("prettyForm('x + y',0,0)", "'x + y'")["correct"])
        self.assertFalse(self.normalized("prettyForm('x + y',0,0)", "'x+y'")["correct"])
        self.assertFalse(self.normalized("prettyForm('x + y',0,0)", "x+y")["correct"])
        for mode in ("strict", "relaxed"):  # Without the rule the display text must match exactly.
            example = dict(RE2, output=json.dumps(reference))
            self.assertFalse(score_prediction(example, "<prediction>{'Q.integer', 'Q.positive'}</prediction>", mode)["correct"])
            self.assertTrue(score_prediction(example, f"<prediction>{reference!r}</prediction>", mode)["correct"])

    def test_keywords_are_parameters_positionals_are_content(self):
        # A repr omits parameters at their defaults: extra keywords in the prediction are allowed,
        # keywords in the reference must match, positional arguments must match exactly.
        for prediction in ("GaussianNB()", "GaussianNB(var_smoothing=100)", "'GaussianNB(priors=None)'"):
            self.assertTrue(self.normalized("GaussianNB()", prediction)["correct"])
        for prediction in ("GaussianNB(123)", "BernoulliNB()", "GaussianNB", "other.GaussianNB()"):
            self.assertFalse(self.normalized("GaussianNB()", prediction)["correct"])
        self.assertTrue(self.normalized("pkg.Class()", "pkg.Class(x=1)")["correct"])
        self.assertTrue(self.normalized("RBF(length_scale=0.112)", "RBF(length_scale=0.112, length_scale_bounds=(1e-05, 100000.0))")["correct"])
        self.assertFalse(self.normalized("RBF(length_scale=0.112)", "RBF(length_scale=0.2)")["correct"])
        self.assertFalse(self.normalized("RBF(length_scale=0.112)", "RBF()")["correct"])
        self.assertTrue(self.normalized("RFE(estimator=SVC(kernel='linear'), step=0.1)",
                                        "RFE(step=0.1, estimator=SVC(kernel='linear', C=1.0), verbose=0)")["correct"])
        self.assertFalse(self.normalized("RFE(estimator=SVC(kernel='linear'), step=0.1)",
                                         "RFE(step=0.1, estimator=SVC(kernel='rbf'))")["correct"])
        for reference, prediction in (("set()", "set([1])"), ("Counter()", "Counter({'a': 1})"),
                                      ("frozenset({1})", "frozenset({1, 2})"), ("slice(0, 1, None)", "slice(0, 1)")):
            self.assertFalse(self.normalized(reference, prediction)["correct"])
        for reference, prediction in (("frozenset({1})", "frozenset({1})"), ("slice(0, 1, None)", "slice(0,1,None)"),
                                      ("Matrix([[1, 0], [0, 1]])", "Matrix([[1,0],[0,1]])"), ("np.uint8(2)", "np.uint8(2)")):
            self.assertTrue(self.normalized(reference, prediction)["correct"])
        self.assertFalse(self.normalized("Matrix([[1, 0], [0, 1]])", "Matrix([[1.0, 0], [0, 1]])")["correct"])
        # Python builtins and standard collections take content, not defaults.
        for reference, prediction in (("Counter()", "Counter(a=3)"), ("dict()", "dict(a=1)"), ("set()", "set(a=1)"),
                                      ("collections.Counter()", "collections.Counter(a=3)"),
                                      ("dict(a=1)", "dict(a=1, b=2)"), ("SimpleNamespace(a=1)", "SimpleNamespace(a=1, b=2)"),
                                      ("defaultdict(int)", "defaultdict(int, a=1)")):
            self.assertFalse(self.normalized(reference, prediction)["correct"], (reference, prediction))
        for reference, prediction in (("dict(a=1, b=2)", "dict(b=2, a=1)"), ("Counter(a=3)", "Counter(a=3)"),
                                      ("OrderedDict()", "OrderedDict()")):
            self.assertTrue(self.normalized(reference, prediction)["correct"], (reference, prediction))
        self.assertFalse(self.normalized("Foo(a=1)", "Foo(**kw)")["correct"])
        example = dict(RE2, output='"GaussianNB()"')
        for mode in ("strict", "relaxed"):
            self.assertFalse(score_prediction(example, "<prediction>GaussianNB(var_smoothing=100)</prediction>", mode)["correct"])
        self.assertTrue(score_prediction(example, "<prediction>GaussianNB()</prediction>", "relaxed")["correct"])

    def test_expressions_compare_as_syntax_trees(self):
        self.assertTrue(self.normalized("Equivalent(Q.even(X*Z), Q.even(X) | Q.even(Z))",
                                        "Equivalent(Q.even(X * Z), (Q.even(X) | Q.even(Z)))")["correct"])
        self.assertTrue(self.normalized("ceiling(x) >= x", "ceiling(x)>=x")["correct"])
        self.assertTrue(self.normalized("-2*(cos(phi))", "-2*cos(phi)")["correct"])
        self.assertTrue(self.normalized("GridSpec(1, 1)[0:1, 0:1]", "GridSpec(1, 1)[0:1, 0:1]")["correct"])
        self.assertFalse(self.normalized("ceiling(x) >= x", "ceiling(x) > x")["correct"])
        self.assertFalse(self.normalized("2 - 2*cos(phi)", "-2*cos(phi) + 2")["correct"])  # syntax, not algebra
        # A plain value stored as text keeps its type distinction from strict.
        self.assertFalse(self.normalized("4", "4")["correct"])
        self.assertTrue(self.normalized("4", "'4'")["correct"])
        self.assertFalse(self.normalized("True", "True")["correct"])
        # Literals JSON could not store compare by value.
        self.assertTrue(self.normalized("(1, 2)", "(1, 2)")["correct"])
        self.assertFalse(self.normalized("(1, 2)", "[1, 2]")["correct"])
        self.assertTrue(self.normalized("b'\x00'", "b'\x00'")["correct"])

    def test_preserve_nested_types(self):
        self.assertFalse(same_value([True], [1]))
        self.assertFalse(same_value((1,), [1]))
        self.assertFalse(same_value({True: 2}, {1: 2}))
        self.assertTrue(same_value({"a": [1, "x"]}, {"a": [1, "x"]}))
        self.assertFalse(same_value({True}, {1}))

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

    def test_non_finite_values(self):
        example = dict(RE2, output='{"score": -Infinity}')
        for prediction in ("{'score': -inf}", "{'score': float('-inf')}", "{'score': -float('inf')}"):
            self.assertTrue(score_prediction(example, f"<prediction>{prediction}</prediction>")["correct"], prediction)
        self.assertFalse(score_prediction(example, "<prediction>{'score': inf}</prediction>")["correct"])
        self.assertTrue(score_prediction(dict(RE2, output="NaN"), "<prediction>nan</prediction>")["correct"])
        self.assertTrue(score_prediction(dict(CRUX, output="float('inf')"), "<prediction>inf</prediction>")["correct"])
        for prediction in ("math.inf", "np.nan", "float('x')", "Infinity()"):
            self.assertEqual(score_prediction(example, f"<prediction>{prediction}</prediction>")["status"],
                             "invalid_prediction", prediction)

    def test_object_strings_compare_exactly_below_normalized(self):
        example = dict(RE2, output='"array([1])"')
        self.assertTrue(score_prediction(example, "<prediction>'array([1])'</prediction>")["correct"])
        self.assertFalse(score_prediction(example, "<prediction>'array([1.])'</prediction>")["correct"])
        self.assertFalse(score_prediction(example, "<prediction>array([1])</prediction>")["correct"])  # unquoted
        self.assertTrue(score_prediction(example, "<prediction>array([1])</prediction>", "relaxed")["correct"])
        self.assertFalse(score_prediction(example, "<prediction>array([1.])</prediction>", "relaxed")["correct"])
        self.assertTrue(score_prediction(example, "<prediction>array([1.])</prediction>", "normalized")["correct"])
        plain = score_prediction(dict(RE2, output='"4"'), "<prediction>4</prediction>")
        self.assertEqual((plain["status"], plain["correct"]), ("scored", False))

    def test_relaxed_accepts_unquoted_strings_only(self):
        for value in ("Yes", "^a/?$", "   "):
            example = dict(CRUX, output=repr(value))
            text = f"<prediction>{value}</prediction>"
            self.assertFalse(score_prediction(example, text)["correct"])
            self.assertTrue(score_prediction(example, text, "relaxed")["correct"])
        self.assertFalse(score_prediction(CRUX, "<prediction>wrong</prediction>", "relaxed")["correct"])
        self.assertFalse(score_prediction(dict(CRUX, output="'4'"),
                                         "<prediction>4</prediction>", "relaxed")["correct"])
        self.assertEqual(score_prediction(CRUX, "x", "relaxed")["status"], "invalid_response")

    def test_review_regressions(self):
        for reference, prediction in [("set()", "set([1])"), ("Counter()", "Counter({'a':1})")]:
            self.assertFalse(self.normalized(reference, prediction)["correct"])
        for value in ("<div>a</div>", "Integral(x, x)", "[ True False]"):
            for mode in SCORING_MODES:
                result = score_prediction(dict(RE2, output=json.dumps(value)),
                                          f"<prediction>{value!r}</prediction>", mode)
                self.assertTrue(result["correct"])
        for mode in ("relaxed", "normalized"):
            for value in ("Yes", "hello world", "   "):
                result = score_prediction(dict(RE2, output=json.dumps(value)),
                                          f"<prediction>\n{value}\n</prediction>", mode)
                self.assertTrue(result["correct"])
        self.assertFalse(self.normalized("['a' 'b']", "array(['a','b'])")["correct"])  # string arrays: exact text only
        self.assertTrue(self.normalized(["array([1,2])"], "[array([1,2])]")["correct"])
        self.assertEqual(self.normalized(5, "five")["status"], "invalid_prediction")


if __name__ == "__main__":
    unittest.main()

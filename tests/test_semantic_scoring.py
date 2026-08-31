import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "analysis"))

from semantic_scoring import (
    SemanticStatus,
    is_truncated_response,
    recursively_typed_equal,
    score_python_literal,
)


class SemanticScoringTest(unittest.TestCase):
    def test_typed_match_accepts_serialization_only_difference(self):
        score = score_python_literal("[1,2]", "[1, 2]")
        self.assertFalse(score.strict_correct)
        self.assertTrue(score.typed_semantic_correct)
        self.assertEqual(score.semantic_status, SemanticStatus.TYPED_MATCH.value)

    def test_recursive_type_identity_rejects_bool_as_int(self):
        self.assertFalse(recursively_typed_equal(True, 1))
        self.assertFalse(recursively_typed_equal({True: "x"}, {1: "x"}))
        self.assertFalse(recursively_typed_equal([1], [True]))

    def test_sequence_and_mapping_types_are_preserved(self):
        self.assertFalse(score_python_literal("(1, 2)", "[1, 2]").typed_semantic_correct)
        self.assertTrue(
            score_python_literal("{'b': 2, 'a': 1}", "{'a': 1, 'b': 2}").typed_semantic_correct
        )

    def test_bare_string_is_separate_exploratory_policy(self):
        score = score_python_literal("hello", "'hello'")
        self.assertFalse(score.typed_semantic_correct)
        self.assertTrue(score.bare_string_content_correct)
        self.assertEqual(score.semantic_status, SemanticStatus.BARE_STRING_MATCH.value)

    def test_empty_and_parse_errors_are_distinct(self):
        empty = score_python_literal("", "1")
        malformed = score_python_literal("[1", "[1]")
        self.assertEqual(empty.semantic_status, SemanticStatus.EMPTY_PREDICTION.value)
        self.assertEqual(malformed.semantic_status, SemanticStatus.PREDICTION_PARSE_ERROR.value)

    def test_truncation_uses_both_finish_reason_fields(self):
        self.assertTrue(is_truncated_response("length", ""))
        self.assertTrue(is_truncated_response("", "max_output_tokens"))
        self.assertFalse(is_truncated_response("stop", "completed"))
        score = score_python_literal("", "1", finish_reason="length")
        self.assertTrue(score.is_truncated)
        self.assertFalse(score.has_extracted_answer)


if __name__ == "__main__":
    unittest.main()


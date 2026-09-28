import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from program_properties import executable_lines, measure_size


class SizeTests(unittest.TestCase):
    def test_comments_and_docstrings_excluded(self):
        source = '# comment\n"""module\ndoc"""\n\ndef f(x):\n    """function doc"""\n    y = "# value"  # comment\n    return y\n'
        self.assertEqual(executable_lines(source, 'cruxeval')['executable_lines'], 3)

    def test_multiline_and_semicolons(self):
        source = 'x = (\n    1 +\n    2\n)\ny = 3; z = 4\n'
        self.assertEqual(executable_lines(source, 'cruxeval')['executable_lines'], 5)

    def test_assigned_multiline_string_retained(self):
        source = 'value = """a\nb\n"""\n'
        self.assertEqual(executable_lines(source, 'cruxeval')['executable_lines'], 3)

    def test_helper_label_only_for_re2(self):
        source = 'pkg.module.helper\n\ndef helper():\n    return 1\n'
        result = executable_lines(source, 're2')
        self.assertEqual(result['executable_lines'], 2)
        self.assertEqual(result['helper_labels_excluded'], 1)
        self.assertEqual(executable_lines(source, 'cruxeval')['executable_lines'], 3)

    def test_re2_label_variants(self):
        # Dot-prefixed, hyphenated package before a decorator, and a comprehension frame before its fragment.
        source = ('.pkg.module.helper\n\ndef helper():\n    return 1\n'
                  'scikit-learn.sklearn.utils.check\n@decorator\ndef check():\n    return 2\n'
                  'pkg.module.<genexpr>\nvalues = [x for x in range(3)]\n')
        result = executable_lines(source, 're2')
        self.assertEqual((result['helper_labels_excluded'], result['line_status']), (3, 'ok'))
        self.assertEqual(result['executable_lines'], 6)

    def test_dotted_name_before_code_is_kept(self):
        result = executable_lines('pkg.module.value\nx = 1\n', 're2')
        self.assertEqual((result['helper_labels_excluded'], result['executable_lines']), (0, 2))

    def test_failure_is_not_zero(self):
        result = executable_lines('x = (\n', 're2')
        self.assertIsNone(result['executable_lines'])
        self.assertEqual(result['line_status'], 'tokenization_failed')
        self.assertEqual(executable_lines('# only a comment\n', 're2')['executable_lines'], 0)

    def test_lexical_count_on_incomplete_function(self):
        result = executable_lines('def f():\n', 're2')
        self.assertEqual(result['line_status'], 'lexical_only')
        self.assertEqual(result['executable_lines'], 1)

    def test_tokenize_exact_fields_only(self):
        class RecordingEncoding:
            def __init__(self): self.seen = []
            def encode_ordinary(self, text):
                self.seen.append(text)
                return list(text)
        encoder = RecordingEncoding()
        measure_size({'dataset': 'cruxeval', 'code': '# comment\nx=1\n', 'input': ' 1 ', 'output': 'secret'}, encoder)
        self.assertEqual(encoder.seen, ['# comment\nx=1\n', ' 1 '])


if __name__ == '__main__':
    unittest.main()

import math
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from program_properties import executable_lines, measure_complexity, measure_size


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


def complexity(code, entrypoint='f', dataset='cruxeval'):
    result = measure_complexity({'dataset': dataset, 'code': code, 'entrypoint': entrypoint})
    return result['cyclomatic_complexity'], result['target_cyclomatic_complexity'], result['complexity_target']


class ComplexityTests(unittest.TestCase):
    def test_straight_line_code_is_one(self):
        self.assertEqual(complexity('def f(x):\n    return x + 1\n'), (1, 1, 'f'))

    def test_decision_points(self):
        # if, elif, conditional expression, while, for, and/or (2), comprehension loop and condition, except.
        source = ('def f(x):\n    if x:\n        pass\n    elif x > 1:\n        pass\n    else:\n        pass\n'
                  '    y = 1 if x else 2\n    while x:\n        x -= 1\n    for i in range(3):\n        pass\n'
                  '    z = x and y or i\n    w = [v for v in range(3) if v]\n'
                  '    try:\n        pass\n    except ValueError:\n        pass\n    return z\n')
        self.assertEqual(complexity(source), (11, 11, 'f'))

    def test_match_cases(self):
        source = 'def f(x):\n    match x:\n        case 1:\n            return 1\n        case _:\n            return 0\n'
        self.assertEqual(complexity(source), (3, 3, 'f'))

    def test_target_excludes_helpers(self):
        source = 'def helper(x):\n    return 1 if x else 0\n\ndef f(x):\n    return helper(x)\n'
        self.assertEqual(complexity(source), (2, 1, 'f'))

    def test_script_without_entrypoint_is_its_own_target(self):
        self.assertEqual(complexity('x = input()\nif x:\n    print(x)\n', entrypoint=''), (2, 2, '<module>'))

    def test_method_in_class_wins_over_top_level_copies(self):
        # RE2 lists helper methods as top-level copies, labelled with their dotted path.
        source = ('pkg.Other.eq\ndef eq(self, other):\n    return self is other\n\n'
                  'class Basic:\n    def eq(self, other):\n        if other is None:\n            return False\n        return True\n')
        self.assertEqual(complexity(source, entrypoint='eq', dataset='re2'), (2, 2, 'Basic.eq'))

    def test_ambiguous_target_has_no_target_complexity(self):
        source = 'def f():\n    return 1\n\ndef g():\n    def f():\n        return 2 if True else 3\n    return f()\n'
        self.assertEqual(complexity(source), (2, None, ''))

    def test_unparsed_code_has_no_complexity(self):
        self.assertEqual(complexity('def f(:\n'), (None, None, ''))


def halstead(code, entrypoint='f'):
    result = measure_complexity({'dataset': 'cruxeval', 'code': code, 'entrypoint': entrypoint})
    return tuple(result[k] for k in ('halstead_volume', 'halstead_difficulty',
                                     'target_halstead_volume', 'target_halstead_difficulty'))


class HalsteadTests(unittest.TestCase):
    def test_operators_and_operands(self):
        # Operators '=', '+' (2 distinct, 2 uses); operands x, a, 1 (3 distinct, 3 uses).
        volume, difficulty, _, _ = halstead('x = a + 1\n', entrypoint='')
        self.assertEqual((volume, difficulty), (round(5 * math.log2(5), 2), 1.0))

    def test_keywords_are_operators_and_layout_is_skipped(self):
        # Operators def ( ) : return (5); operands f, x, x (2 distinct, 3 uses).
        self.assertEqual(halstead('def f(x):\n    return x\n')[:2], (round(8 * math.log2(7), 2), 3.75))

    def test_comments_skipped_and_docstring_is_one_operand(self):
        plain = halstead('def f(x):\n    return x\n')
        self.assertEqual(halstead('def f(x):  # note\n    return x\n'), plain)
        # The docstring adds one distinct operand: f, x, x, '"""doc"""' -> 3 distinct, 4 uses.
        self.assertEqual(halstead('def f(x):\n    """doc"""\n    return x\n')[1], round(5 / 2 * 4 / 3, 2))

    def test_target_uses_only_its_lines(self):
        source = 'def helper(x):\n    return x * 2 + 1\n\ndef f(x):\n    return x\n'
        self.assertEqual(halstead(source)[2:], halstead('def f(x):\n    return x\n')[:2])

    def test_unparsed_code_has_no_halstead(self):
        self.assertEqual(halstead('def f(:\n'), (None, None, None, None))


def flow(code, entrypoint='f'):
    return measure_complexity({'dataset': 'cruxeval', 'code': code, 'entrypoint': entrypoint})['information_flow']


class InformationFlowTests(unittest.TestCase):
    def test_fan_in_times_fan_out_squared(self):
        # Fan-in 4: parameter x, outside name LIMIT, attributes base and items read (self is not counted).
        # Fan-out 3: calls to helper and len, attribute total written.
        source = ('class C:\n    def f(self, x):\n        self.total = helper(x) + self.base + LIMIT\n'
                  '        return len(self.items)\n')
        self.assertEqual(flow(source), (4 * 3) ** 2)

    def test_local_names_and_methods(self):
        # Fan-in 1 (nums); the local output is not read from outside. Fan-out 2: methods append and count.
        source = 'def f(nums):\n    output = []\n    for n in nums:\n        output.append(nums.count(n))\n    return output\n'
        self.assertEqual(flow(source), (1 * 2) ** 2)

    def test_no_calls_or_writes_is_zero(self):
        self.assertEqual(flow('def f(x):\n    return x + 1\n'), 0)

    def test_script_and_unidentified_target(self):
        self.assertEqual(flow('s = input()\nprint(s)\n', entrypoint=''), (0 * 2) ** 2)
        self.assertIsNone(flow('def f():\n    return 1\n\ndef g():\n    def f():\n        return 2\n    return f()\n'))


if __name__ == '__main__':
    unittest.main()

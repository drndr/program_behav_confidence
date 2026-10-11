import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from execution import trace
from measure_execution import decoded, harness, measure, output_matches, run_program, same_output

LOOP = "def f(n):\n    total = 0\n    for i in range(n):\n        total += i\n    return total\n"


class TraceTests(unittest.TestCase):
    def test_counts_each_executed_line(self):
        outcome = trace("def f(x):\n    y = x + 1\n    z = y * 2\n    return z\n", "f(1)")
        self.assertEqual((outcome["trace_length"], outcome["result"]), (3, "4"))

    def test_loop_body_counts_once_per_iteration(self):
        # One more iteration adds the for line and the body line.
        self.assertEqual(trace(LOOP, "f(5)")["trace_length"] - trace(LOOP, "f(4)")["trace_length"], 2)

    def test_definitions_and_setup_are_not_traced(self):
        outcome = trace("x = 1\ny = 2\ndef f():\n    return x + y + z\n", "f()", setup="z = 3")
        self.assertEqual((outcome["trace_length"], outcome["result"]), (1, "6"))

    def test_script_reads_stdin_and_output_is_captured(self):
        outcome = trace("s = input()\nprint(s * 2)\n", stdin="ab\n")
        self.assertEqual((outcome["trace_length"], outcome["printed"]), (2, "abab\n"))


class RunProgramTests(unittest.TestCase):
    def test_child_process_matches_in_process_trace(self):
        outcome = run_program({"code": LOOP, "call": "f(3)"}, timeout=30)
        self.assertEqual((outcome["status"], outcome["trace_length"], outcome["result"]),
                         ("ok", trace(LOOP, "f(3)")["trace_length"], "3"))

    def test_timeout_and_error(self):
        self.assertEqual(run_program({"code": "def f():\n    while True:\n        pass\n", "call": "f()"}, 2)["status"],
                         "timeout")
        outcome = run_program({"code": "def f():\n    return 1 / 0\n", "call": "f()"}, timeout=30)
        self.assertEqual(outcome["status"], "error")
        self.assertIn("ZeroDivisionError", outcome["detail"])

    def test_fixed_hash_seed_makes_set_order_repeatable(self):
        payload = {"code": "def f():\n    return list({'alpha', 'beta', 'gamma', 'delta', 'epsilon'})\n", "call": "f()"}
        self.assertEqual(run_program(payload, 30)["result"], run_program(payload, 30)["result"])


def re2(item_id, code, entrypoint, recorded, output):
    return {"dataset": "re2", "id": item_id, "code": code, "entrypoint": entrypoint,
            "input": json.dumps(recorded), "output": json.dumps(output)}


class HarnessTests(unittest.TestCase):
    def test_cruxeval_calls_the_entrypoint_with_its_input(self):
        example = {"dataset": "cruxeval", "code": "def f(a, b):\n    return a\n", "entrypoint": "f", "input": "1, 2"}
        self.assertEqual(harness(example), [({"code": example["code"], "call": "f(1, 2)"}, False)])

    def test_re2_harness_per_source(self):
        function = re2("HumanEval_1", "def g(x):\n    return x\n", "g", {"self": {}, "args": {"x": "7"}, "kwargs": {}}, 7)
        self.assertEqual([p["call"] for p, _ in harness(function)], ["g(**{'x': '7'}, **{})", "g(**{'x': 7}, **{})"])
        method = re2("ClassEval_1@Box.get", "class Box:\n    def get(self):\n        return self.v\n", "get",
                     {"self": {"v": 1}, "args": {}, "kwargs": {}}, 1)
        (payload, _), = harness(method)  # nothing to decode, so one candidate
        self.assertEqual(payload["setup"], "_self = Box.__new__(Box)\n_self.__dict__.update({'v': 1})\n")
        script = re2("atcoder_ABC1_A", "print(input())\n", "", {"input": "hi"}, "hi")
        self.assertEqual(harness(script), [({"code": "print(input())\n", "stdin": "hi\n"}, False)])
        self.assertEqual(harness(re2("pkg@@mod.py@@f_L1", "", "f", {}, None)), [])

    def test_decoding_keeps_strings_that_stay_strings(self):
        self.assertEqual(decoded({"a": "(1-2j)", "b": ["3", "'x'", "plain"]}), {"a": 1 - 2j, "b": [3, "'x'", "plain"]})

    def test_output_compared_as_values(self):
        self.assertTrue(same_output("[1, 2]", "[1,2]"))
        self.assertFalse(same_output("'a'", "'b'"))
        self.assertTrue(same_output("<object>", "<object>"))
        tuple_result = {"status": "ok", "result_json": json.dumps([1, "(2+0j)"]), "printed": ""}
        self.assertTrue(output_matches(re2("HumanEval_2", "", "g", {}, [1, "(2+0j)"]), tuple_result))


class MeasureTests(unittest.TestCase):
    def test_decoded_attempt_kept_when_it_reproduces_the_reference(self):
        example = re2("ClassEval_3@C.m", "class C:\n    @staticmethod\n    def m(a, b):\n        return a * b\n", "m",
                      {"self": {}, "args": {"a": "(1+1j)", "b": "(1-1j)"}, "kwargs": {}}, "(2+0j)")
        row = measure(example, timeout=30)
        self.assertEqual((row["execution_status"], row["output_matches"], row["decoded_inputs"], row["trace_length"]),
                         ("ok", True, True, 1))

    def test_avatar_script_compares_printed_output(self):
        example = re2("atcoder_ABC2_A", "a, b = map(int, input().split())\nprint(a + b)\n", "", {"input": "2 3"}, 5)
        row = measure(example, timeout=30)
        self.assertEqual((row["output_matches"], row["trace_length"]), (True, 2))


if __name__ == "__main__":
    unittest.main()

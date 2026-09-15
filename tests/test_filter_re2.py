from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from filter_re2 import remove_prompts


def row(case_id, task, prompt_input, example=None):
    text = (example or "") + task + f"\n[INPUT]\n```json\n{prompt_input}\n```\n[/INPUT]\n"
    return {"id": case_id, "messages": [{"role": "user", "content": text}], "target_output": "1"}


NAMED = ("[PYTHON]\ndef target(a):\n    return helper(a)\n[/PYTHON]\n"
         "Functions called during the execution:\n[PYTHON]\ndef helper(a):\n    return a\n[/PYTHON]\n"
         "What will be the output of `target`, given the following input:")
STDIN = "[PYTHON]\nprint(input())\n[/PYTHON]\nWhat will be the output of the following input:"
EXAMPLE = ("[PYTHON]\ndef example():\n    return 0\n[/PYTHON]\nWhat will be the output of `example`, "
           "given the following input:\n[INPUT]\n{}\n[/INPUT]\n[/EXAMPLE]\n")


class RemovePromptsTests(unittest.TestCase):
    def test_named_entrypoint_and_helper_code(self):
        cleaned = remove_prompts([row("a", NAMED, '{"self": {}, "args": {"a": 1}, "kwargs": {}}', EXAMPLE)])[0]
        self.assertEqual(cleaned["entrypoint"], "target")
        self.assertEqual(cleaned["code"], "def target(a):\n    return helper(a)\n\ndef helper(a):\n    return a")
        self.assertEqual(cleaned["input"], '{"self": {}, "args": {"a": 1}, "kwargs": {}}')

    def test_stdin_program_has_empty_entrypoint(self):
        cleaned = remove_prompts([row("b", STDIN, '{"input": "3"}')])[0]
        self.assertEqual((cleaned["entrypoint"], cleaned["code"]), ("", "print(input())"))

    def test_task_question_wins_over_worked_example(self):
        # Prompts without an [/EXAMPLE] boundary still name the task's own function last.
        example = EXAMPLE.replace("[/EXAMPLE]\n", "")
        cleaned = remove_prompts([row("c", NAMED, '{"self": {}, "args": {"a": 1}, "kwargs": {}}', example)])[0]
        self.assertEqual(cleaned["entrypoint"], "target")

    def test_rejects_inconsistent_prompts(self):
        undefined = NAMED.replace("`target`", "`missing`")
        with self.assertRaisesRegex(ValueError, "not defined"):
            remove_prompts([row("d", undefined, "{}")])
        with self.assertRaisesRegex(ValueError, "stdin input only"):
            remove_prompts([row("e", STDIN, '{"args": {}}')])
        with self.assertRaisesRegex(ValueError, "task question"):
            remove_prompts([row("f", "[PYTHON]\nx = 1\n[/PYTHON]\nNo question here.", "{}")])


if __name__ == "__main__":
    unittest.main()

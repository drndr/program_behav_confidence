import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


splitter = load_module(
    "freeze_re2_calibration_transfer_split",
    REPO_ROOT / "scripts" / "analysis" / "freeze_re2_calibration_transfer_split.py",
)


class Re2TransferSplitTest(unittest.TestCase):
    def test_split_is_exact_deterministic_and_preserves_ids(self):
        rows = []
        for index in range(10):
            rows.append(
                {
                    "id": f"sample_{index}",
                    "complexity_group": "HC" if index < 5 else "LC",
                    "source_dataset": "Real",
                    "safe_compatibility_correct": index % 2 == 0,
                    "matched_132_subset": index < 4,
                }
            )

        first = splitter.assign_split(rows, seed=7)
        second = splitter.assign_split(rows, seed=7)

        self.assertEqual(first, second)
        self.assertEqual(set(first), {row["id"] for row in rows})
        self.assertEqual(sum(value == "calibration" for value in first.values()), 5)
        self.assertEqual(sum(value == "test" for value in first.values()), 5)


if __name__ == "__main__":
    unittest.main()

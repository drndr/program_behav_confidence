from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze import property_groups, thirds


class SizeThirdsTests(unittest.TestCase):
    def test_thirds_partition_items_and_keep_ties_together(self):
        sizes = [1, 2, 2, 2, 3, 4, 5, 6, 6, 9]
        rows = [{"dataset": "cruxeval", "code_tokens": size, "executable_lines": None} for size in sizes]
        groups = [g for g in property_groups(rows) if g[2] == "code_tokens"]
        members = [[row["code_tokens"] for row in rows if member(row)] for *_, member in groups]
        self.assertEqual(sorted(sum(members, [])), sizes)
        self.assertTrue(all(members))
        for value in set(sizes):
            self.assertEqual(sum(value in third for third in members), 1, value)
        self.assertEqual([round(bound, 2) for bound in thirds(sizes)[0]], [1, 2.0])

    def test_skips_properties_without_enough_distinct_values(self):
        rows = [{"dataset": "re2", "code_tokens": 5, "executable_lines": 3} for _ in range(6)]
        self.assertEqual(property_groups(rows), [])


if __name__ == "__main__":
    unittest.main()

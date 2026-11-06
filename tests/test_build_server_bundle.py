import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_server_bundle import INPUTS, NOTEBOOK, build


class ServerBundleTests(unittest.TestCase):
    @unittest.skipUnless((ROOT / ".git").exists(), "needs a git checkout")
    def test_files_keep_repository_paths_and_recorded_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "bundle.zip"
            manifest = build(out)
            with zipfile.ZipFile(out) as archive:
                names = set(archive.namelist())
                self.assertEqual(names, {f"bundle/{path}" for path in INPUTS}
                                 | {"bundle/qwen38_27b_transformers.ipynb", "bundle/README.md", "bundle/bundle_manifest.json"})
                self.assertEqual(json.loads(archive.read("bundle/bundle_manifest.json")), manifest)
                for path, digest in manifest["files"].items():
                    self.assertEqual(hashlib.sha256(archive.read(f"bundle/{path}")).hexdigest(), digest)

    def test_notebook_code_cells_parse(self):
        notebook = json.loads((ROOT / NOTEBOOK).read_text(encoding="utf-8"))
        cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertTrue(cells)
        for cell in cells:
            ast.parse("".join(cell["source"]))


if __name__ == "__main__":
    unittest.main()

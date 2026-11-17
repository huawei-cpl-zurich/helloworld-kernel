"""Verify imported source pins without importing device packages."""
import ast
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1] / 'catalog/matmul/catlass'


class CatlassMatmulCatalog(unittest.TestCase):
    def test_source_and_license_pins(self):
        provenance = json.loads((ROOT / 'provenance.json').read_text())
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(digest(ROOT / 'kernel.py'),
                         provenance['extraction']['extracted_file_sha256'])
        self.assertEqual(digest(ROOT / 'LICENSE'), provenance['license_sha256'])
        self.assertEqual(digest(ROOT / 'helpers/params.py'),
                         next(iter(provenance['dependencies'].values())))
        tree = ast.parse((ROOT / 'kernel.py').read_text())
        functions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
        self.assertIn('basic_mmad_kernel', functions)

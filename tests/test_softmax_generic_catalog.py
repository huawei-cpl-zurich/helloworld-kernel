"""Generic upstream softmax functions and dependencies remain verbatim."""
import ast
import hashlib
import json
from pathlib import Path
import unittest
ROOT = Path(__file__).resolve().parents[1] / 'catalog/softmax/triton-generic'


class GenericSoftmaxCatalog(unittest.TestCase):
    def test_source_helper_and_license(self):
        p = json.loads((ROOT / 'provenance.json').read_text())
        ast.parse((ROOT / 'kernel.py').read_bytes())
        for name, expected in [('kernel.py', p['extraction']['extracted_file_sha256']),
                               ('extension.py', p['dependency_extractions']['extension.py']['extracted_file_sha256']),
                               ('LICENSE', p['license_sha256'])]:
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected)

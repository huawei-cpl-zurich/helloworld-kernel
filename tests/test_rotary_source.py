"""Exact original GPU functions and licensing pins."""
import ast
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1] / 'catalog/rotary_embedding/triton'


class RotarySource(unittest.TestCase):
    def test_original_source_and_called_helper(self):
        provenance = json.loads((ROOT / 'provenance.json').read_text())
        for name, expected in [('kernel.py', provenance['extraction']['extracted_file_sha256']),
                               ('LICENSE', provenance['license_sha256'])]:
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected)
        tree = ast.parse((ROOT / 'kernel.py').read_bytes())
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        self.assertEqual({n.name for n in nodes},
                         {'rotary_embedding_rw_kernel', 'rotary_embedding_siso_kernel'})
        signature = ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False)
        self.assertEqual(hashlib.sha256(signature.encode()).hexdigest(),
                         provenance['extraction']['gpu_ast_sha256'])

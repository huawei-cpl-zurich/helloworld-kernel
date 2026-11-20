"""CPU reference semantics and pinned upstream source; no device claims."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1] / 'catalog/matmul'


class MatmulCatalog(unittest.TestCase):
    def test_reference_uses_float_accumulation_and_preserves_output_type(self):
        spec = importlib.util.spec_from_file_location('matmul_reference', ROOT / 'spec/reference.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        a = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float16)
        b = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=torch.float16)
        result = module.reference(a, b)
        self.assertEqual(set(result), {'out'})
        self.assertEqual(result['out'].dtype, torch.float16)
        self.assertTrue(torch.equal(result['out'], b[:2]))
        self.assertTrue(torch.equal(a, torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=a.dtype)))

    def test_upstream_source_and_license_are_pinned(self):
        provenance = json.loads((ROOT / 'triton/provenance.json').read_text())
        for name, pin in [('kernel.py', 'source_sha256'), ('LICENSE', 'license_sha256')]:
            self.assertEqual(hashlib.sha256((ROOT / 'triton' / name).read_bytes()).hexdigest(), provenance[pin])
        tree = ast.parse((ROOT / 'triton/kernel.py').read_bytes())
        functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        self.assertTrue({'matmul_kernel', 'leaky_relu_custom', 'get_autotune_config'} <= functions)

"""Source provenance and pure CPU oracle checks."""
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parents[1] / 'catalog/softmax'


class SoftmaxCatalog(unittest.TestCase):
    def test_source_and_license_pins(self):
        provenance = json.loads((ROOT / 'triton/provenance.json').read_text())
        for name, pin in [('kernel.py', 'source_sha256'), ('LICENSE', 'license_sha256')]:
            self.assertEqual(hashlib.sha256((ROOT / 'triton' / name).read_bytes()).hexdigest(), provenance[pin])
        tree = ast.parse((ROOT / 'triton/kernel.py').read_bytes())
        kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'softmax_kernel')
        if sys.version_info[:2] == (3, 13):
            digest = hashlib.sha256(ast.dump(kernel, include_attributes=False).encode()).hexdigest()
            self.assertEqual(digest, provenance['kernel_ast_sha256'])

    def test_stable_reference_and_input_preservation(self):
        spec = importlib.util.spec_from_file_location('softmax_reference', ROOT / 'spec/reference.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        x = torch.tensor([[10000, 10000, 10000], [-10000, -10000, -10000]], dtype=torch.float16)
        saved = x.clone()
        out = module.reference(x)['out']
        self.assertEqual(out.dtype, x.dtype)
        self.assertTrue(torch.equal(x, saved))
        self.assertTrue(torch.allclose(out.float(), torch.full((2, 3), 1 / 3), atol=0.001))
        self.assertTrue(torch.isfinite(out).all())

    def test_reference_preserves_nonuniform_rows(self):
        spec = importlib.util.spec_from_file_location('softmax_reference', ROOT / 'spec/reference.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        logits = torch.tensor([[0, 1, 2]], dtype=torch.float16)
        normalizer = 1 + math.e + math.e**2
        expected = torch.tensor([[1, math.e, math.e**2]], dtype=torch.float32) / normalizer
        self.assertTrue(torch.allclose(module.reference(logits)['out'].float(), expected, atol=0.001))

    def test_factory_binds_both_frozen_shapes(self):
        triton = ModuleType('triton')
        language = ModuleType('triton.language')
        triton.language = language
        language.constexpr = object()
        triton.jit = lambda function: function
        triton.next_power_of_2 = lambda size: 1 << (size - 1).bit_length()
        spec = importlib.util.spec_from_file_location('softmax_factory', ROOT / 'triton/factory.py')
        factory = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'triton': triton, 'triton.language': language}):
            spec.loader.exec_module(factory)
            for rows, columns, block in ((32, 128, 128), (45, 257, 512)):
                shape = {'shape': [rows, columns]}
                result = factory.build({'inputs': [shape], 'outputs': [shape], 'scalars': []})
                self.assertEqual(result['bindings'], [
                    ['output_ptr', 'output', 'out'], ['input_ptr', 'input', 'x'],
                ])
                self.assertEqual(result['constants'], {
                    'input_row_stride': columns, 'output_row_stride': columns,
                    'n_rows': rows, 'n_cols': columns, 'BLOCK_SIZE': block,
                })
                self.assertEqual(result['grid'], [32, 1, 1])

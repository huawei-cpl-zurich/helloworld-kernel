"""Original source pins and CPU SwiGLU semantics."""
import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parents[1] / 'catalog/swiglu'


class SwigluCatalog(unittest.TestCase):
    def test_source_and_license_pins(self):
        provenance = json.loads((ROOT / 'triton/provenance.json').read_text())
        for name, pin in [('kernel.py', 'source_sha256'), ('LICENSE', 'license_sha256')]:
            self.assertEqual(hashlib.sha256((ROOT / 'triton' / name).read_bytes()).hexdigest(), provenance[pin])
        license_pin = provenance['additional_license']
        self.assertEqual(hashlib.sha256((ROOT / 'triton' / license_pin['path']).read_bytes()).hexdigest(),
                         license_pin['sha256'])
        tree = ast.parse((ROOT / 'triton/kernel.py').read_bytes())
        names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        self.assertTrue({'silu', '_swiglu_forward_kernel'} <= names)

    def test_gate_order_and_input_preservation(self):
        spec = importlib.util.spec_from_file_location('swiglu_reference', ROOT / 'spec/reference.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        a = torch.tensor([[-1, 0, 1]], dtype=torch.float16)
        b = torch.tensor([[2, 3, 4]], dtype=torch.float16)
        saved = (a.clone(), b.clone())
        out = module.reference(a, b)['out']
        expected = (a.float() / (1 + torch.exp(-a.float())) * b.float()).half()
        self.assertTrue(torch.equal(out, expected))
        self.assertTrue(torch.equal(a, saved[0]) and torch.equal(b, saved[1]))
        self.assertEqual(out.dtype, a.dtype)

    def test_factory_matches_both_frozen_shapes(self):
        triton = ModuleType('triton')
        language = ModuleType('triton.language')
        triton.language = language
        triton.jit = lambda function: function
        triton.next_power_of_2 = lambda size: 1 << (size - 1).bit_length()
        language.constexpr = object()
        spec = importlib.util.spec_from_file_location('swiglu_factory', ROOT / 'triton/factory.py')
        factory = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'triton': triton, 'triton.language': language}):
            spec.loader.exec_module(factory)
            for rows, columns, block in ((32, 128, 128), (45, 257, 512)):
                tensor = {'shape': [rows, columns]}
                result = factory.build({'inputs': [tensor, tensor], 'outputs': [tensor], 'scalars': []})
                self.assertIs(result['kernel'].__globals__['silu'], factory.module.silu)
                self.assertEqual(result['bindings'], [
                    ['a_ptr', 'input', 'a'], ['b_ptr', 'input', 'b'], ['c_ptr', 'output', 'out'],
                ])
                self.assertEqual(result['constants'], {'stride': columns, 'n_cols': columns, 'BLOCK_SIZE': block})
                self.assertEqual(result['grid'], [rows, 1, 1])
                self.assertEqual(list(inspect.signature(result['kernel']).parameters),
                                 ['a_ptr', 'b_ptr', 'c_ptr', 'stride', 'n_cols', 'BLOCK_SIZE'])

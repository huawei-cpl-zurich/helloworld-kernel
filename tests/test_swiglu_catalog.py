"""Original source pins and CPU SwiGLU semantics."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1] / 'catalog/swiglu'


class SwigluCatalog(unittest.TestCase):
    def test_source_and_license_pins(self):
        provenance = json.loads((ROOT / 'triton/provenance.json').read_text())
        for name, pin in [('kernel.py', 'source_sha256'), ('LICENSE', 'license_sha256')]:
            self.assertEqual(hashlib.sha256((ROOT / 'triton' / name).read_bytes()).hexdigest(), provenance[pin])
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

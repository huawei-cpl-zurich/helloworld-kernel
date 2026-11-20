"""Source pins and independent CPU normalization checks."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1] / 'catalog/layer_norm'


class LayerNormCatalog(unittest.TestCase):
    def test_source_and_license_pins(self):
        provenance = json.loads((ROOT / 'triton/provenance.json').read_text())
        for name, pin in [('kernel.py', 'source_sha256'), ('LICENSE', 'license_sha256')]:
            self.assertEqual(hashlib.sha256((ROOT / 'triton' / name).read_bytes()).hexdigest(), provenance[pin])
        tree = ast.parse((ROOT / 'triton/kernel.py').read_bytes())
        self.assertIn('_layer_norm_fwd_fused', [n.name for n in tree.body if isinstance(n, ast.FunctionDef)])

    def test_affine_normalization_and_constant_row(self):
        spec = importlib.util.spec_from_file_location('layer_norm_reference', ROOT / 'spec/reference.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        x = torch.tensor([[1, 2, 3], [2, 2, 2]], dtype=torch.float16)
        weight = torch.tensor([2, 3, 4], dtype=x.dtype)
        bias = torch.tensor([1, -1, 2], dtype=x.dtype)
        saved = x.clone()
        result = module.reference(x, weight, bias, 1e-5)
        expected = torch.nn.functional.layer_norm(x.float(), (3,), weight.float(), bias.float(), 1e-5).half()
        self.assertTrue(torch.equal(result['out'], expected))
        self.assertTrue(torch.equal(result['out'][1], bias))
        self.assertTrue(torch.equal(result['mean'], torch.tensor([2., 2.])))
        self.assertTrue(torch.allclose(result['rstd'], torch.tensor([(2/3+1e-5)**-0.5, 1e-5**-0.5])))
        self.assertTrue(torch.equal(x, saved))

"""Pinned source and mutable CPU contract checks."""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
import torch
ROOT = Path(__file__).resolve().parents[1] / 'catalog/add_rms_norm'


class AddRmsNormCatalog(unittest.TestCase):
    def test_source_and_license_pins(self):
        provenance = json.loads((ROOT / 'triton/provenance.json').read_text())
        for name, pin in [('kernel.py', 'source_sha256'), ('LICENSE', 'license_sha256')]:
            self.assertEqual(hashlib.sha256((ROOT / 'triton' / name).read_bytes()).hexdigest(), provenance[pin])

    def test_aliases_residual_update_and_weight_preservation(self):
        spec = importlib.util.spec_from_file_location('add_rms_reference', ROOT / 'spec/reference.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        x = torch.tensor([[1, 2]], dtype=torch.float16)
        residual = torch.tensor([[1, 0]], dtype=x.dtype)
        weight = torch.tensor([3, 4], dtype=x.dtype)
        result = module.reference(x, residual, weight, 0.)
        self.assertIs(result['out'], x)
        self.assertIs(result['residual_out'], residual)
        self.assertTrue(torch.equal(x, torch.tensor([[3, 4]], dtype=x.dtype)))
        self.assertTrue(torch.equal(residual, torch.tensor([[2, 2]], dtype=x.dtype)))
        self.assertTrue(torch.equal(weight, torch.tensor([3, 4], dtype=x.dtype)))

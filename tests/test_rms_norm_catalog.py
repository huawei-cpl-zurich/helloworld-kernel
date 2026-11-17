"""Exact source pins and analytic RMS normalization oracle."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import unittest
import torch
ROOT = Path(__file__).resolve().parents[1] / 'catalog/rms_norm'


class RmsNormCatalog(unittest.TestCase):
    def test_source_and_license_pins(self):
        provenance = json.loads((ROOT / 'triton/provenance.json').read_text())
        for name, expected in [('kernel.py', provenance['extraction']['extracted_file_sha256']),
                               ('LICENSE', provenance['license_sha256'])]:
            self.assertEqual(hashlib.sha256((ROOT / 'triton' / name).read_bytes()).hexdigest(), expected)

    def test_reciprocal_rms_affine_weight_and_zero_row(self):
        spec = importlib.util.spec_from_file_location('rms_reference', ROOT / 'spec/reference.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        x = torch.tensor([[3, 4]], dtype=torch.float16)
        gamma = torch.tensor([1, 2], dtype=x.dtype)
        out = module.reference(x, gamma, epsilon=0.)
        self.assertTrue(torch.equal(out['y'], torch.tensor([[3/math.sqrt(12.5), 8/math.sqrt(12.5)]], dtype=x.dtype)))
        self.assertAlmostEqual(float(out['rstd'][0]), 1/math.sqrt(12.5), places=6)
        self.assertTrue(torch.equal(x, torch.tensor([[3, 4]], dtype=x.dtype)))
        zero = module.reference(torch.zeros_like(x), gamma, epsilon=1e-4)
        self.assertTrue(torch.equal(zero['y'], torch.zeros_like(x)))
        self.assertEqual(float(zero['rstd'][0]), 100.)

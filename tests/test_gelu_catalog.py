"""Original source pins and analytic in-place GELU checks."""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
import torch
ROOT = Path(__file__).resolve().parents[1] / 'catalog/gelu'


class GeluCatalog(unittest.TestCase):
    def test_source_and_license_pins(self):
        provenance = json.loads((ROOT / 'triton/provenance.json').read_text())
        for name, pin in [('kernel.py', 'source_sha256'), ('LICENSE', 'license_sha256')]:
            self.assertEqual(hashlib.sha256((ROOT / 'triton' / name).read_bytes()).hexdigest(), provenance[pin])

    def test_alias_zero_and_symmetric_identity_in_both_modes(self):
        spec = importlib.util.spec_from_file_location('gelu_reference', ROOT / 'spec/reference.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for mode in (False, True):
            x = torch.tensor([-1, 0, 1], dtype=torch.float16)
            out = module.reference(x, mode)['out']
            self.assertIs(out, x)
            self.assertEqual(float(out[1]), 0.)
            self.assertAlmostEqual(float(out[2] - out[0]), 1., places=3)
            self.assertGreater(float(out[2]), 0.8)
            self.assertLess(float(out[0]), 0.)

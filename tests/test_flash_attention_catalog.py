"""Source pins and analytic causal attention oracle checks."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import unittest
import torch
ROOT = Path(__file__).resolve().parents[1] / 'catalog/flash_attention_prefill'


class FlashAttentionCatalog(unittest.TestCase):
    def test_original_source_and_license(self):
        provenance = json.loads((ROOT / 'triton/provenance.json').read_text())
        for name, expected in [('kernel.py', provenance['extraction']['extracted_file_sha256']),
                               ('LICENSE', provenance['license_sha256'])]:
            self.assertEqual(hashlib.sha256((ROOT / 'triton' / name).read_bytes()).hexdigest(), expected)

    def test_causal_prefix_mean_natural_lse_and_device(self):
        spec = importlib.util.spec_from_file_location('flash_reference', ROOT / 'spec/reference.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        q = torch.zeros((1, 1, 2, 2), dtype=torch.float16)
        v = torch.tensor([[[[2, 4], [6, 8]]]], dtype=q.dtype)
        out = module.reference(q, q.clone(), v, 0.125)
        self.assertTrue(torch.equal(out['out'], torch.tensor([[[[2, 4], [4, 6]]]], dtype=q.dtype)))
        self.assertTrue(torch.allclose(out['lse'], torch.tensor([[[0., math.log(2.)]]])))
        meta = torch.empty((1, 1, 2, 2), dtype=q.dtype, device='meta')
        self.assertTrue(all(t.device.type == 'meta' for t in module.reference(meta, meta, meta, 0.125).values()))

    def test_v2_preserves_inputs_and_reference_and_versions_tolerance(self):
        old = (ROOT / 'spec/cannery.yaml').read_text()
        new = (ROOT / 'spec-v2/cannery.yaml').read_text()
        self.assertIn('  version: 1\n', old)
        self.assertIn('default_atol: 0.0001', old)
        self.assertIn('  version: 2\n', new)
        self.assertIn('default_atol: 0.001', new)
        self.assertIn('default_rtol: 0.001', new)
        for name in ('reference.py', 'recipes.yaml', 'tensor-contract.yaml', 'environment.lock'):
            self.assertEqual((ROOT / 'spec' / name).read_bytes(),
                             (ROOT / 'spec-v2' / name).read_bytes())
        expected = old.replace('  version: 1\n', '  version: 2\n').replace(
            'default_atol: 0.0001', 'default_atol: 0.001')
        self.assertEqual(expected, new)

"""Original GPU source pins and grouped CPU oracle."""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
import torch
ROOT = Path(__file__).resolve().parents[1] / 'catalog/grouped_matmul'


class GroupedMatmulCatalog(unittest.TestCase):
    def test_original_source_and_license(self):
        provenance = json.loads((ROOT / 'triton/provenance.json').read_text())
        for name, expected in [('kernel.py', provenance['extraction']['extracted_file_sha256']),
                               ('LICENSE', provenance['license_sha256'])]:
            self.assertEqual(hashlib.sha256((ROOT / 'triton' / name).read_bytes()).hexdigest(), expected)

    def test_cumulative_group_ends_and_packed_weights(self):
        spec = importlib.util.spec_from_file_location('grouped_reference', ROOT / 'spec/reference.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        a = torch.tensor([[1], [2], [3]], dtype=torch.float16)
        b = torch.tensor([[2], [5]], dtype=a.dtype)
        out = module.reference(a, b, torch.tensor([2, 3], dtype=torch.int32))['out']
        self.assertTrue(torch.equal(out, torch.tensor([[2], [4], [15]], dtype=a.dtype)))
        self.assertTrue(torch.equal(a, torch.tensor([[1], [2], [3]], dtype=a.dtype)))

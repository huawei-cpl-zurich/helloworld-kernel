"""Source pins and analytic paged KV lookup."""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
import torch
ROOT = Path(__file__).resolve().parents[1] / 'catalog/paged_attention_decode'


class PagedAttentionCatalog(unittest.TestCase):
    def test_source_and_license_pins(self):
        provenance = json.loads((ROOT / 'triton/provenance.json').read_text())
        for name, expected in [('kernel.py', provenance['extraction']['extracted_file_sha256']),
                               ('LICENSE', provenance['license_sha256'])]:
            self.assertEqual(hashlib.sha256((ROOT / 'triton' / name).read_bytes()).hexdigest(), expected)

    def test_page_table_order_length_and_grouped_heads(self):
        spec = importlib.util.spec_from_file_location('paged_reference', ROOT / 'spec/reference.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        q = torch.zeros((1, 2, 2), dtype=torch.float16)
        k = torch.zeros((2, 2, 1, 2), dtype=q.dtype)
        v = torch.tensor([[[[10, 20]], [[30, 40]]], [[[2, 4]], [[6, 8]]]], dtype=q.dtype)
        out = module.reference(q, k, v, torch.tensor([1, 0]), torch.tensor([0]),
                               torch.tensor([1]), torch.tensor([3]), 0.5)['out']
        self.assertTrue(torch.equal(out, torch.tensor([[[6, 32 / 3], [6, 32 / 3]]], dtype=q.dtype)))

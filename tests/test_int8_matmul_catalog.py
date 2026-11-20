"""Original source pins and independently calculated scaled INT8 products."""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
import torch
ROOT = Path(__file__).resolve().parents[1] / 'catalog/int8_dequant_matmul'


class Int8MatmulCatalog(unittest.TestCase):
    def test_source_and_license_pins(self):
        provenance = json.loads((ROOT / 'triton/provenance.json').read_text())
        for name, expected in [('kernel.py', provenance['extraction']['extracted_file_sha256']),
                               ('LICENSE', provenance['license_sha256'])]:
            self.assertEqual(hashlib.sha256((ROOT / 'triton' / name).read_bytes()).hexdigest(), expected)

    def test_signed_products_and_row_column_scales(self):
        spec = importlib.util.spec_from_file_location('int8_reference', ROOT / 'spec/reference.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        a = torch.tensor([[1, -2], [3, 4]], dtype=torch.int8)
        b = torch.tensor([[5, 6], [-7, 8]], dtype=torch.int8)
        out = module.reference(a, b, torch.tensor([0.5, 2.]), torch.tensor([4., 0.25]))['out']
        self.assertTrue(torch.equal(out, torch.tensor([[38., -1.25], [-104., 25.]], dtype=torch.float16)))
        self.assertTrue(torch.equal(a, torch.tensor([[1, -2], [3, 4]], dtype=a.dtype)))

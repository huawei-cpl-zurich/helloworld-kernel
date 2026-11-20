"""Pinned original GPU/helper source and analytic CPU semantics."""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
import torch
ROOT = Path(__file__).resolve().parents[1] / 'catalog/embedding'


class Catalog(unittest.TestCase):
    def test_original_source_helper_and_license(self):
        provenance = json.loads((ROOT / 'triton/provenance.json').read_text())
        pins = [('kernel.py', provenance['extraction']['extracted_file_sha256']),
                ('extension.py', provenance['dependency_extractions']['extension.py']['extracted_file_sha256']),
                ('LICENSE', provenance['license_sha256'])]
        for name, expected in pins:
            self.assertEqual(hashlib.sha256((ROOT / 'triton' / name).read_bytes()).hexdigest(), expected)

    def test_cpu_semantics_and_readonly_inputs(self):
        spec = importlib.util.spec_from_file_location('catalog_reference', ROOT / 'spec/reference.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        weight = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float16)
        saved = weight.clone()
        out = module.reference(weight, torch.tensor([2, 0, 2], dtype=torch.int64))['out']
        self.assertTrue(torch.equal(out, torch.tensor([[5, 6], [1, 2], [5, 6]], dtype=weight.dtype)))
        self.assertTrue(torch.equal(weight, saved))

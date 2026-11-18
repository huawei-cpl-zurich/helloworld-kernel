"""Pinned original GPU source and analytic token-index semantics."""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
import torch
ROOT = Path(__file__).resolve().parents[1] / 'catalog/moe_token_unpermute'


class TokenCatalog(unittest.TestCase):
    def test_original_source_and_license(self):
        provenance = json.loads((ROOT / 'triton/provenance.json').read_text())
        for name, expected in [('kernel.py', provenance['extraction']['extracted_file_sha256']),
                               ('LICENSE', provenance['license_sha256'])]:
            self.assertEqual(hashlib.sha256((ROOT / 'triton' / name).read_bytes()).hexdigest(), expected)

    def test_indices_weights_and_input_preservation(self):
        spec = importlib.util.spec_from_file_location('token_reference', ROOT / 'spec/reference.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        tokens = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.float16)
        before = tokens.clone()
        out = module.reference(tokens, torch.tensor([3, 0, 2, 1]),
                               torch.tensor([[0.25, 0.75], [0.5, 0.5]]))['output']
        self.assertTrue(torch.equal(out, torch.tensor([[2.5, 3.5], [4, 5]], dtype=tokens.dtype)))
        self.assertTrue(torch.equal(tokens, before))

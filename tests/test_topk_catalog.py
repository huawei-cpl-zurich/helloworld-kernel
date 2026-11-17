"""Source dependency closure and analytic sorted selection."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
import torch
ROOT = Path(__file__).resolve().parents[1] / 'catalog/topk'


class TopkCatalog(unittest.TestCase):
    def test_original_source_license_and_helper(self):
        provenance = json.loads((ROOT / 'triton/provenance.json').read_text())
        for name, expected in [('kernel.py', provenance['extraction']['extracted_file_sha256']),
                               ('LICENSE', provenance['license_sha256'])]:
            self.assertEqual(hashlib.sha256((ROOT / 'triton' / name).read_bytes()).hexdigest(), expected)
        tree = ast.parse((ROOT / 'triton/kernel.py').read_bytes())
        self.assertEqual({n.name for n in tree.body if isinstance(n, ast.FunctionDef)},
                         {'_get_finfo_val', 'topk_stage1_kernel'})
        self.assertEqual(len([n for n in tree.body if isinstance(n, ast.AnnAssign)]), 6)

    def test_largest_smallest_values_indices_and_input_preservation(self):
        spec = importlib.util.spec_from_file_location('topk_reference', ROOT / 'spec/reference.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        x = torch.tensor([[4., 1., 3., 2.]])
        for largest, values, indices in [(True, [4., 3.], [0, 2]), (False, [1., 2.], [1, 3])]:
            out = module.reference(x, 2, largest)
            self.assertTrue(torch.equal(out['values'], torch.tensor([values])))
            self.assertTrue(torch.equal(out['indices'], torch.tensor([indices])))
        self.assertTrue(torch.equal(x, torch.tensor([[4., 1., 3., 2.]])))

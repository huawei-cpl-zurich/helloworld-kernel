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
                               ('LICENSE', provenance['license_sha256']),
                               ('extension.py', provenance['extension']['extracted_file_sha256'])]:
            self.assertEqual(hashlib.sha256((ROOT / 'triton' / name).read_bytes()).hexdigest(), expected)
        tree = ast.parse((ROOT / 'triton/kernel.py').read_bytes())
        self.assertEqual({n.name for n in tree.body if isinstance(n, ast.FunctionDef)},
                         {'_get_finfo_val', 'topk_stage1_kernel'})
        self.assertEqual(len([n for n in tree.body if isinstance(n, (ast.AnnAssign, ast.Assign))]), 6)

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

    def test_factory_loads_finfo_and_program_helpers(self):
        import sys
        from types import ModuleType
        from unittest.mock import patch

        class Constant:
            def __init__(self, value):
                self.value = value

        def jit(function=None, **options):
            return jit if function is None else function

        modules = {name: ModuleType(name) for name in ('triton', 'triton.language')}
        modules['triton'].language = modules['triton.language']
        modules['triton'].jit = jit
        modules['triton'].next_power_of_2 = lambda size: 1 << (size - 1).bit_length()
        modules['triton.language'].constexpr = Constant
        modules['triton.language'].tensor = object()
        spec = importlib.util.spec_from_file_location('catalog_topk_factory', ROOT / 'triton/factory.py')
        factory = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(factory)
            self.assertIsInstance(factory.module._MIN_FLOAT32_VAL, Constant)
            self.assertEqual(factory.module._MIN_FLOAT32_VAL.value, torch.finfo(torch.float32).min)
            self.assertIsInstance(factory.module._MAX_FLOAT16_VAL, Constant)
            self.assertEqual(factory.module._MAX_FLOAT16_VAL.value, torch.finfo(torch.float16).max)
            for rows, columns, block in ((32, 128, 128), (45, 257, 512)):
                result = factory.build({
                    'inputs': [{'shape': [rows, columns]}],
                    'scalars': [{'name': 'k', 'value': 8}, {'name': 'largest', 'value': True}],
                })
                helpers = result['kernel'].__globals__['tle']
                self.assertIs(helpers.program_id, factory.ext.program_id)
                self.assertIs(helpers.num_programs, factory.ext.num_programs)
                self.assertEqual(result['bindings'], [
                    ['y_ptr', 'output', 'values'], ['index_ptr', 'output', 'indices'],
                    ['x_ptr', 'input', 'x'],
                ])
                self.assertEqual(result['grid'], [rows, 1, 1])
                self.assertEqual(result['constants'], {
                    'k': 8, 'N': columns, 'CHUNK_SIZE': block, 'DESCENDING': True,
                })

"""Pinned source and mutable CPU contract checks."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch
import torch
import yaml
ROOT = Path(__file__).resolve().parents[1] / 'catalog/add_rms_norm'


class AddRmsNormCatalog(unittest.TestCase):
    def test_source_and_license_pins(self):
        provenance = json.loads((ROOT / 'triton/provenance.json').read_text())
        for name, pin in [('kernel.py', 'source_sha256'), ('LICENSE', 'license_sha256')]:
            self.assertEqual(hashlib.sha256((ROOT / 'triton' / name).read_bytes()).hexdigest(), provenance[pin])

    def test_aliases_residual_update_and_weight_preservation(self):
        spec = importlib.util.spec_from_file_location('add_rms_reference', ROOT / 'spec/reference.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        x = torch.tensor([[1, 2]], dtype=torch.float16)
        residual = torch.tensor([[1, 2]], dtype=x.dtype)
        weight = torch.tensor([3, 4], dtype=x.dtype)
        result = module.reference(x, residual, weight, 6.)
        self.assertIs(result['out'], x)
        self.assertIs(result['residual_out'], residual)
        self.assertTrue(torch.equal(x, torch.tensor([[1.5, 4]], dtype=x.dtype)))
        self.assertTrue(torch.equal(residual, torch.tensor([[2, 4]], dtype=x.dtype)))
        self.assertTrue(torch.equal(weight, torch.tensor([3, 4], dtype=x.dtype)))

    def test_factory_maps_both_frozen_aliased_cases(self):
        triton = ModuleType('triton')
        language = ModuleType('triton.language')
        triton.language = language
        triton.jit = lambda **options: lambda function: function
        triton.next_power_of_2 = lambda size: 1 << (size - 1).bit_length()
        language.constexpr = object()
        spec = importlib.util.spec_from_file_location('add_rms_factory', ROOT / 'triton/factory.py')
        factory = importlib.util.module_from_spec(spec)
        manifest = yaml.safe_load((ROOT / 'spec/cannery.yaml').read_text())['spec']
        recipes = yaml.safe_load((ROOT / 'spec/recipes.yaml').read_text())['cases']
        contract = yaml.safe_load((ROOT / 'spec/tensor-contract.yaml').read_text())['cases']
        with patch.dict(sys.modules, {'triton': triton, 'triton.language': language}):
            spec.loader.exec_module(factory)
            for recipe in recipes:
                case = contract[recipe['id']]
                self.assertEqual(case['aliases'], {'out': 'x', 'residual_out': 'residual'})
                inputs = [{'name': name, **recipe['inputs'][name]} for name in manifest['interface']['inputs']]
                epsilon = case['scalars']['epsilon']
                result = factory.build({
                    'inputs': inputs, 'scalars': [{'name': 'epsilon', **epsilon}],
                })
                rows, cols = recipe['inputs']['x']['shape']
                self.assertIs(result['kernel'], factory.module.fused_add_rms_norm_kernel)
                self.assertEqual(result['bindings'], [
                    ['X', 'input', 'x'], ['R', 'input', 'residual'], ['W', 'input', 'weight'],
                ])
                self.assertEqual(result['constants'], {
                    'x_stride_r': cols, 'x_stride_c': 1, 'r_stride_r': cols, 'r_stride_c': 1,
                    'N': cols, 'eps': epsilon['value'], 'BLOCK_SIZE': triton.next_power_of_2(cols),
                })
                self.assertEqual(result['grid'], [rows, 1, 1])

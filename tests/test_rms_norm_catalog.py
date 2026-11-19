"""Exact source pins and analytic RMS normalization oracle."""
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch
import torch
ROOT = Path(__file__).resolve().parents[1] / 'catalog/rms_norm'


class RmsNormCatalog(unittest.TestCase):
    def test_source_and_license_pins(self):
        provenance = json.loads((ROOT / 'triton/provenance.json').read_text())
        for name, expected in [('kernel.py', provenance['extraction']['extracted_file_sha256']),
                               ('LICENSE', provenance['license_sha256']),
                               ('LICENSE.liger', provenance['additional_license']['sha256'])]:
            self.assertEqual(hashlib.sha256((ROOT / 'triton' / name).read_bytes()).hexdigest(), expected)

    def test_reciprocal_rms_affine_weight_and_zero_row(self):
        spec = importlib.util.spec_from_file_location('rms_reference', ROOT / 'spec/reference.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        x = torch.tensor([[3, 4]], dtype=torch.float16)
        gamma = torch.tensor([1, 2], dtype=x.dtype)
        out = module.reference(x, gamma, epsilon=0.)
        self.assertTrue(torch.equal(out['y'], torch.tensor([[3/math.sqrt(12.5), 8/math.sqrt(12.5)]], dtype=x.dtype)))
        self.assertAlmostEqual(float(out['rstd'][0]), 1/math.sqrt(12.5), places=6)
        self.assertTrue(torch.equal(x, torch.tensor([[3, 4]], dtype=x.dtype)))
        zero = module.reference(torch.zeros_like(x), gamma, epsilon=1e-4)
        self.assertTrue(torch.equal(zero['y'], torch.zeros_like(x)))
        self.assertEqual(float(zero['rstd'][0]), 100.)

    def test_factory_covers_each_frozen_signature(self):
        triton = ModuleType('triton')
        language = ModuleType('triton.language')
        tl_math = ModuleType('triton.language.math')
        triton.language = language
        language.math = tl_math
        tl_math.rsqrt = object()
        language.constexpr = int
        triton.jit = lambda function: function
        triton.next_power_of_2 = lambda size: 1 << (size - 1).bit_length()
        modules = {'triton': triton, 'triton.language': language, 'triton.language.math': tl_math}
        spec = importlib.util.spec_from_file_location('rms_factory', ROOT / 'triton/factory.py')
        factory = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(factory)
            for columns in (128, 512):
                result = factory.build({
                    'inputs': [{'shape': [32, columns]}, {'shape': [columns]}],
                    'scalars': [{'name': 'epsilon', 'value': 1e-5}],
                })
                self.assertIs(result['kernel'].__globals__['rsqrt'], tl_math.rsqrt)
                self.assertEqual(result['bindings'], [
                    ['Y_ptr', 'output', 'y'], ['X_ptr', 'input', 'x'],
                    ['W_ptr', 'input', 'gamma'], ['RSTD_ptr', 'output', 'rstd'],
                ])
                self.assertEqual(result['constants'], {
                    'Y_row_stride': columns, 'X_row_stride': columns, 'W_row_stride': 0,
                    'RSTD_row_stride': 1, 'n_cols': columns, 'eps': 1e-5, 'offset': 0.0,
                    'casting_mode': 1, 'BLOCK_SIZE': columns,
                })
                bound = {binding[0] for binding in result['bindings']} | result['constants'].keys()
                self.assertEqual(set(inspect.signature(result['kernel']).parameters), bound)
                self.assertEqual(result['grid'], [32, 1, 1])

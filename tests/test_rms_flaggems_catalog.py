"""Original alternative RMSNorm GPU/license source pins."""
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

from rms_norm_cases import signatures

ROOT = Path(__file__).resolve().parents[1] / 'catalog/rms_norm/triton-flaggems'


class RmsFlaggemsCatalog(unittest.TestCase):
    def test_verbatim_source_and_license(self):
        p = json.loads((ROOT / 'provenance.json').read_text())
        for name, expected in [('kernel.py', p['extraction']['extracted_file_sha256']),
                               ('LICENSE', p['license_sha256'])]:
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected)

    def test_factory_covers_each_frozen_signature(self):
        triton = ModuleType('triton')
        language = ModuleType('triton.language')
        triton.language = language
        language.constexpr = object()

        def jit(**options):
            def decorate(function):
                function.arg_names = list(inspect.signature(function).parameters)
                return function
            return decorate

        triton.jit = jit
        spec = importlib.util.spec_from_file_location('flaggems_rms_factory', ROOT / 'factory.py')
        factory = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'triton': triton, 'triton.language': language}):
            spec.loader.exec_module(factory)
            for variant in signatures():
                inputs = {tensor['name']: tensor for tensor in variant['inputs']}
                rows, columns = inputs['x']['shape']
                epsilon = next(scalar['value'] for scalar in variant['scalars'] if scalar['name'] == 'epsilon')
                result = factory.build(variant)
                self.assertIs(result['kernel'].__globals__['tl'], language)
                self.assertEqual(result['bindings'], [
                    ['out_ptr', 'output', 'y'], ['INV_RMS', 'output', 'rstd'],
                    ['in_ptr', 'input', 'x'], ['w_ptr', 'input', 'gamma'],
                ])
                self.assertEqual(result['constants'], {
                    'y_stride_r': columns, 'y_stride_c': 1, 'x_stride_r': columns,
                    'x_stride_c': 1, 'N': columns, 'eps': epsilon, 'BLOCK_SIZE': 4096,
                })
                bound = {binding[0] for binding in result['bindings']} | result['constants'].keys()
                self.assertEqual(set(inspect.signature(result['kernel']).parameters), bound)
                self.assertEqual(result['grid'], [rows, 1, 1])
            with self.assertRaisesRegex(ValueError, 'at most 4096'):
                factory.build({'inputs': [{'shape': [32, 4097]}, {'shape': [4097]}]})

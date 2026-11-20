"""Original Liger forward GPU/helper and BSD license pins."""
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1] / 'catalog/swiglu/triton-liger'


class LigerSourceCatalog(unittest.TestCase):
    def test_original_gpu_and_license(self):
        p = json.loads((ROOT / 'provenance.json').read_text())
        for name, expected in [('kernel.py', p['extraction']['extracted_file_sha256']),
                               ('LICENSE', p['license_sha256'])]:
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected)

    def test_factory_matches_both_frozen_shapes(self):
        triton = ModuleType('triton')
        language = ModuleType('triton.language')
        triton.language = language
        triton.jit = lambda function: function
        language.constexpr = object()
        spec = importlib.util.spec_from_file_location('liger_factory', ROOT / 'factory.py')
        factory = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'triton': triton, 'triton.language': language}):
            spec.loader.exec_module(factory)
            for rows, columns in ((32, 128), (45, 257)):
                tensor = {'shape': [rows, columns]}
                result = factory.build({'inputs': [tensor, tensor], 'outputs': [tensor], 'scalars': []})
                self.assertIs(result['kernel'].__globals__['original'], factory.original)
                self.assertIs(factory.original._swiglu_forward_kernel.__globals__['silu'],
                              factory.original.silu)
                self.assertEqual(result['bindings'], [
                    ['a_ptr', 'input', 'a'], ['b_ptr', 'input', 'b'], ['c_ptr', 'output', 'out'],
                ])
                self.assertEqual(result['constants'], {'stride': columns, 'n_cols': columns, 'BLOCK_SIZE': 4096})
                self.assertEqual(result['grid'], [rows, 1, 1])
                self.assertEqual(list(inspect.signature(result['kernel']).parameters),
                                 ['a_ptr', 'b_ptr', 'c_ptr', 'stride', 'n_cols', 'BLOCK_SIZE'])
            oversized = {'shape': [32, 4097]}
            with self.assertRaisesRegex(ValueError, 'at most 4096'):
                factory.build({'inputs': [oversized, oversized], 'outputs': [oversized], 'scalars': []})

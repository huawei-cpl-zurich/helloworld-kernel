"""Original source pins and analytic in-place GELU checks."""
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch
import torch
import yaml
ROOT = Path(__file__).resolve().parents[1] / 'catalog/gelu'


class GeluCatalog(unittest.TestCase):
    def test_source_and_license_pins(self):
        provenance = json.loads((ROOT / 'triton/provenance.json').read_text())
        for name, pin in [('kernel.py', 'source_sha256'), ('LICENSE', 'license_sha256')]:
            self.assertEqual(hashlib.sha256((ROOT / 'triton' / name).read_bytes()).hexdigest(), provenance[pin])

    def test_alias_zero_and_symmetric_identity_in_both_modes(self):
        spec = importlib.util.spec_from_file_location('gelu_reference', ROOT / 'spec/reference.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for mode in (False, True):
            x = torch.tensor([-1, 0, 1], dtype=torch.float16)
            out = module.reference(x, mode)['out']
            self.assertIs(out, x)
            self.assertEqual(float(out[1]), 0.)
            self.assertAlmostEqual(float(out[2] - out[0]), 1., places=3)
            self.assertGreater(float(out[2]), 0.8)
            self.assertLess(float(out[0]), 0.)

    def test_factory_uses_the_frozen_alias_and_tiling(self):
        modules = {}
        for name in ('triton', 'triton.language', 'triton.language.extra',
                     'triton.language.extra.cann', 'triton.language.extra.cann.extension'):
            module = modules[name] = ModuleType(name)
            parent, _, attr = name.rpartition('.')
            if parent:
                setattr(modules[parent], attr, module)
        modules['triton'].jit = lambda function: function
        modules['triton.language'].constexpr = object()
        spec = importlib.util.spec_from_file_location('gelu_factory', ROOT / 'triton/factory.py')
        factory = importlib.util.module_from_spec(spec)
        manifest = yaml.safe_load((ROOT / 'spec/cannery.yaml').read_text())['spec']
        recipes = yaml.safe_load((ROOT / 'spec' / manifest['cases']['recipe_file']).read_text())
        contract = yaml.safe_load((ROOT / 'spec' / manifest['cases']['tensor_contract_file']).read_text())
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(factory)
            for case in recipes['cases']:
                case_contract = contract['cases'][case['id']]
                input_name, = manifest['interface']['inputs']
                output_name, = manifest['interface']['outputs']
                self.assertEqual(case_contract['aliases'][output_name], input_name)
                scalar = case_contract['scalars']['approximate_tanh']
                shape = case['inputs'][input_name]['shape']
                count, = shape
                result = factory.build({
                    'inputs': [{'name': input_name, **case['inputs'][input_name]}],
                    'scalars': [{'name': 'approximate_tanh', **scalar}],
                })
                self.assertIs(result['kernel'], factory.module.gelu_)
                self.assertEqual(result['bindings'], [['x_ptr', 'input', input_name]])
                self.assertEqual(result['constants'], {
                    'n_elements': count, 'USE_TANH': scalar['value'], 'BLOCK_SIZE': 4096,
                })
                self.assertEqual(result['grid'], [(count + 4095) // 4096, 1, 1])
                bound = {binding[0] for binding in result['bindings']} | result['constants'].keys()
                self.assertEqual(set(inspect.signature(result['kernel']).parameters), bound)

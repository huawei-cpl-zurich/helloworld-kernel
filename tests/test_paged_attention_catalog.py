"""Source pins and analytic paged KV lookup."""
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import torch
import yaml
ROOT = Path(__file__).resolve().parents[1] / 'catalog/paged_attention_decode'


class PagedAttentionCatalog(unittest.TestCase):
    def test_source_and_license_pins(self):
        provenance = json.loads((ROOT / 'triton/provenance.json').read_text())
        for name, expected in [('kernel.py', provenance['extraction']['extracted_file_sha256']),
                               ('LICENSE', provenance['license_sha256']),
                               ('LICENSE.lmdeploy', provenance['additional_license']['sha256'])]:
            self.assertEqual(hashlib.sha256((ROOT / 'triton' / name).read_bytes()).hexdigest(), expected)

    def test_page_table_order_length_and_grouped_heads(self):
        spec = importlib.util.spec_from_file_location('paged_reference', ROOT / 'spec/reference.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        q = torch.zeros((2, 4, 2), dtype=torch.float16)
        q[..., 0] = 1
        k = torch.zeros((2, 2, 2, 2), dtype=q.dtype)
        k[0, 0, :, 0] = 2
        first_head = torch.tensor([[[[10, 20]], [[30, 40]]], [[[2, 4]], [[6, 8]]]], dtype=q.dtype)
        v = torch.cat([first_head, first_head + torch.tensor([10, 20])], dim=2)
        k = torch.cat([k, k])
        v = torch.cat([v, v])
        out = module.reference(
            q, k, v, torch.tensor([1, 0, 2, 3]), torch.tensor([0, 1]),
            torch.tensor([1, 1]), torch.tensor([3, 2]), math.log(2) / 2,
        )['out']
        self.assertTrue(torch.equal(out, torch.tensor([
            [[7, 13], [7, 13], [17, 33], [17, 33]],
            [[50 / 3, 80 / 3], [50 / 3, 80 / 3], [80 / 3, 140 / 3], [80 / 3, 140 / 3]],
        ], dtype=q.dtype)))

    def test_factory_maps_each_frozen_paged_variant(self):
        triton = ModuleType('triton')
        language = ModuleType('triton.language')
        triton.language = language
        triton.jit = lambda function: function
        language.constexpr = object()
        language.math = SimpleNamespace(exp=object(), fdiv=object())
        loader = importlib.util.spec_from_file_location('paged_factory', ROOT / 'triton/factory.py')
        factory = importlib.util.module_from_spec(loader)
        manifest = yaml.safe_load((ROOT / 'spec/cannery.yaml').read_text())['spec']
        recipes = yaml.safe_load((ROOT / 'spec/recipes.yaml').read_text())['cases']
        contract = yaml.safe_load((ROOT / 'spec/tensor-contract.yaml').read_text())['cases']
        with patch.dict(sys.modules, {'triton': triton, 'triton.language': language}):
            loader.loader.exec_module(factory)
            for recipe in recipes:
                cases = recipe['inputs']
                scale = contract[recipe['id']]['scalars']['sm_scale']
                launch = factory.build({
                    'inputs': [{'name': name, **cases[name]} for name in manifest['interface']['inputs']],
                    'scalars': [{'name': 'sm_scale', **scale}],
                })
                rows, heads, dim = cases['q']['shape']
                pages, block, kv_heads, _ = cases['k_cache']['shape']
                self.assertEqual(launch['bindings'], [
                    ['Q', 'input', 'q'], ['K', 'input', 'k_cache'], ['V', 'input', 'v_cache'],
                    ['Q_start_loc', 'input', 'q_start'], ['Q_seqlens', 'input', 'q_lengths'],
                    ['KV_seqlens', 'input', 'kv_lengths'], ['Block_offsets', 'input', 'block_table'],
                    ['Out', 'output', 'out'],
                ])
                bound = {binding[0] for binding in launch['bindings']} | launch['constants'].keys()
                self.assertEqual(set(inspect.signature(launch['kernel']).parameters), bound)
                self.assertIs(launch['kernel'].__globals__['_convert_pv'], factory.module._convert_pv)
                self.assertEqual(launch['grid'], [1, heads, rows])
                self.assertEqual(launch['constants']['stride_boffb'], pages // rows)
                self.assertEqual(launch['constants']['stride_kp'], block * kv_heads * dim)
                self.assertEqual(launch['constants']['kv_group_num'], heads // kv_heads)
                self.assertEqual(launch['constants']['sm_scale'], scale['value'])

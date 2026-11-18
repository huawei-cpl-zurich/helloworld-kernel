"""In-place fused K/V row-scatter semantics and invalid-slot safety."""
import importlib.util
from pathlib import Path
import unittest
import torch
ROOT = Path(__file__).resolve().parents[1] / 'catalog/kv_cache_update'


class KvCacheCatalog(unittest.TestCase):
    def test_invalid_slots_aliases_and_untouched_rows(self):
        spec = importlib.util.spec_from_file_location('kv_reference', ROOT / 'spec/reference.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        updates = torch.arange(16, dtype=torch.float32).reshape(4, 2, 1, 2).to(torch.bfloat16)
        slots = torch.tensor([-1, 1, 3, 99], dtype=torch.int32)
        cache = torch.full((4, 2, 1, 2), -2., dtype=updates.dtype)
        original = updates.clone()
        out = module.reference(updates, slots, cache)['updated']
        self.assertEqual(out.data_ptr(), cache.data_ptr())
        self.assertTrue(torch.equal(out[1], updates[1]))
        self.assertTrue(torch.equal(out[3], updates[2]))
        self.assertTrue(bool((out[[0, 2]] == -2).all()))
        self.assertTrue(torch.equal(updates, original))
        self.assertTrue(torch.equal(slots, torch.tensor([-1, 1, 3, 99], dtype=slots.dtype)))

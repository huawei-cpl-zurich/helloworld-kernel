"""Analytic rotary embedding oracle checks."""
import importlib.util
from pathlib import Path
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1] / 'catalog/rotary_embedding'


class RotaryReference(unittest.TestCase):
    def test_rotation_conventions_and_input_preservation(self):
        spec = importlib.util.spec_from_file_location('rotary_reference', ROOT / 'spec/reference.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        state = torch.tensor([[[1, 2, 3, 4]]], dtype=torch.float16)
        saved = state.clone()
        cos = torch.zeros((1, 2), dtype=state.dtype)
        sin = torch.ones_like(cos)
        for interleaved, expected in [(True, [-2, 1, -4, 3]), (False, [-3, -4, 1, 2])]:
            out = module.reference(state, cos, sin, interleaved)['out']
            self.assertTrue(torch.equal(out, torch.tensor([[expected]], dtype=state.dtype)))
            self.assertTrue(torch.equal(state, saved))
        out = module.reference(state, torch.ones_like(cos), torch.zeros_like(sin), False)['out']
        self.assertTrue(torch.equal(out, state))

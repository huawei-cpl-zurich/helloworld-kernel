"""Analytic signed and zero-row quantization contract."""
import importlib.util
from pathlib import Path
import unittest
import torch
ROOT = Path(__file__).resolve().parents[1] / 'catalog/dynamic_int8_quantize'


class DynamicQuantCatalog(unittest.TestCase):
    def test_signed_values_and_zero_row(self):
        spec = importlib.util.spec_from_file_location('dynamic_quant_reference', ROOT / 'spec/reference.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        x = torch.tensor([[-1., 0.5, 1.], [-2., -0.5, 2.], [0., 0., 0.]], dtype=torch.float16)
        before = x.clone()
        result = module.reference(x)
        self.assertTrue(torch.equal(result['quantized'], torch.tensor([[-127, 64, 127], [-127, -32, 127], [0, 0, 0]], dtype=torch.int8)))
        self.assertTrue(torch.allclose(result['scale'], torch.tensor([1. / 127, 2. / 127, 0.])))
        self.assertTrue(torch.equal(x, before))

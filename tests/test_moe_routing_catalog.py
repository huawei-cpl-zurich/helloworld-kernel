"""Routing semantics, device placement and pinned source."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import unittest
import torch
ROOT = Path(__file__).resolve().parents[1] / 'catalog/moe_routing'


class RoutingCatalog(unittest.TestCase):
    def test_source_and_license_pins(self):
        provenance = json.loads((ROOT / 'triton/provenance.json').read_text())
        for name, pin in [('kernel.py', 'source_sha256'), ('LICENSE', 'license_sha256')]:
            self.assertEqual(hashlib.sha256((ROOT / 'triton' / name).read_bytes()).hexdigest(), provenance[pin])

    def test_expert_order_renormalization_source_rows_and_device(self):
        spec = importlib.util.spec_from_file_location('routing_reference', ROOT / 'spec/reference.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        x = torch.tensor([[0., 1., 2., 3.], [3., 2., 1., 0.]])
        out = module.reference(x, 2, True)
        self.assertTrue(torch.equal(out['experts'], torch.tensor([[3, 2], [0, 1]])))
        self.assertTrue(torch.equal(out['source_rows'], torch.tensor([[0, 2], [1, 3]], dtype=torch.int32)))
        expected = torch.tensor([[1/(1+math.exp(-1)), 1/(1+math.exp(1))]]).expand(2, 2)
        self.assertTrue(torch.allclose(out['weights'], expected))
        self.assertTrue(torch.all(module.reference(x, 2, False)['weights'].sum(-1) < 1))
        self.assertTrue(all(t.device.type == 'meta' for t in module.reference(torch.empty((2, 4), device='meta'), 2, True).values()))
        self.assertTrue(torch.equal(x, torch.tensor([[0., 1., 2., 3.], [3., 2., 1., 0.]])))

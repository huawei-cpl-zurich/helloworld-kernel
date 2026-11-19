"""CPU checks for the MLA decode spec and source pins."""

import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1] / "catalog/mla_decode"


class MlaCatalogTest(unittest.TestCase):
    def test_source_pins(self):
        directory = ROOT / "triton"
        provenance = json.loads((directory / "provenance.json").read_text())
        self.assertEqual(
            hashlib.sha256((directory / "kernel.py").read_bytes()).hexdigest(),
            provenance["kernel_file_sha256"],
        )
        self.assertEqual(
            hashlib.sha256((directory / "LICENSE").read_bytes()).hexdigest(),
            provenance["files"]["LICENSE"],
        )

    def test_reference_respects_page_order_and_ignores_padding_and_rope_values(self):
        spec = importlib.util.spec_from_file_location("mla_reference", ROOT / "spec/reference.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        q = torch.zeros((1, 64, 576), dtype=torch.float16)
        cache = torch.zeros((2, 2, 576), dtype=q.dtype)
        cache[1, :, :512] = 3
        cache[0, 0, :512] = 9
        cache[0, 1, :512] = 99
        cache[..., 512:] = 123
        blocks = torch.tensor([1, 0], dtype=torch.int32)
        lengths = torch.tensor([3], dtype=torch.int32)
        for device in ("cpu", "meta"):
            args = [value.to(device) for value in (q, cache, blocks, lengths)]
            result = module.reference(*args, 1 / 576**0.5)["out"]
            self.assertEqual(result.shape, (1, 64, 512))
            self.assertEqual(result.dtype, q.dtype)
            self.assertEqual(result.device.type, device)
            if device == "cpu":
                self.assertTrue(torch.equal(result, torch.full_like(result, 5)))

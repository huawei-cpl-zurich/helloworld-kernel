"""CPU checks for the MLA decode spec and source pins."""

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

    def test_factory_uses_both_frozen_paged_signatures(self):
        triton = ModuleType("triton")
        language = ModuleType("triton.language")
        triton.language = language
        triton.jit = lambda function: function
        language.constexpr = object()
        language.tensor = object()
        loader = importlib.util.spec_from_file_location("mla_factory", ROOT / "triton/factory.py")
        factory = importlib.util.module_from_spec(loader)
        manifest = yaml.safe_load((ROOT / "spec/cannery.yaml").read_text())["spec"]
        recipes = yaml.safe_load((ROOT / "spec/recipes.yaml").read_text())["cases"]
        contract = yaml.safe_load((ROOT / "spec/tensor-contract.yaml").read_text())["cases"]
        submission = yaml.safe_load((ROOT / "triton/submission.yaml").read_text())
        self.assertEqual(submission["implementation"]["build"]["options"]["triton_compile_mode"], "simd")
        with patch.dict(sys.modules, {"triton": triton, "triton.language": language}):
            loader.loader.exec_module(factory)
            for case in recipes:
                inputs = case["inputs"]
                scale = contract[case["id"]]["scalars"]["sm_scale"]
                launch = factory.build({
                    "inputs": [{"name": name, **inputs[name]} for name in manifest["interface"]["inputs"]],
                    "scalars": [{"name": "sm_scale", **scale}],
                })
                self.assertEqual(launch["bindings"], [
                    ["Q_ptr", "input", "q"],
                    ["Kv_cache", "input", "kv_cache"],
                    ["Req_to_tokens", "input", "block_table"],
                    ["B_seq_len", "input", "kv_lengths"],
                    ["O", "output", "out"],
                ])
                bound = {binding[0] for binding in launch["bindings"]} | launch["constants"].keys()
                self.assertEqual(set(inspect.signature(launch["kernel"]).parameters), bound)
                self.assertIs(launch["kernel"].__globals__["tle"].program_id, factory.module.program_id)
                self.assertEqual(launch["constants"]["stride_req_to_tokens_bs"], inputs["block_table"]["shape"][0])
                self.assertEqual(launch["constants"]["sm_scale"], scale["value"])
                self.assertEqual(launch["grid"], [4, 1, 1])

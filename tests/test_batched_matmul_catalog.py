"""Pinned original GPU/helper source and analytic CPU semantics."""
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
ROOT = Path(__file__).resolve().parents[1] / 'catalog/batched_matmul'


class Catalog(unittest.TestCase):
    def test_original_source_helper_and_license(self):
        provenance = json.loads((ROOT / 'triton/provenance.json').read_text())
        pins = [('kernel.py', provenance['extraction']['extracted_file_sha256']),
                ('extension.py', provenance['dependency_extractions']['extension.py']['extracted_file_sha256']),
                ('LICENSE', provenance['license_sha256'])]
        for name, expected in pins:
            self.assertEqual(hashlib.sha256((ROOT / 'triton' / name).read_bytes()).hexdigest(), expected)

    def test_cpu_semantics_and_readonly_inputs(self):
        spec = importlib.util.spec_from_file_location('catalog_reference', ROOT / 'spec/reference.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        a = torch.tensor([[[1, 2], [3, 4]], [[2, 0], [0, 2]]], dtype=torch.float16)
        b = torch.tensor([[[5, 6], [7, 8]], [[1, 2], [3, 4]]], dtype=a.dtype)
        saved = a.clone()
        out = module.reference(a, b)['out']
        expected = torch.tensor([[[19, 22], [43, 50]], [[2, 4], [6, 8]]], dtype=a.dtype)
        self.assertTrue(torch.equal(out, expected))
        self.assertTrue(torch.equal(a, saved))

    def test_factory_covers_static_kernel_signature(self):
        def jit(function=None, **options):
            if function is None:
                return jit
            function.arg_names = list(inspect.signature(function).parameters)
            return function

        modules = {
            name: ModuleType(name)
            for name in (
                "triton", "triton.language", "triton.language.extra",
                "triton.language.extra.cann", "triton.language.extra.cann.extension",
            )
        }
        for name, module in modules.items():
            if "." in name:
                parent, attribute = name.rsplit(".", 1)
                setattr(modules[parent], attribute, module)
        modules["triton"].jit = jit
        modules["triton.language"].constexpr = object()
        modules["triton.language"].tensor = object()
        modules["triton.language.extra.cann.extension"].constexpr = object()
        spec = importlib.util.spec_from_file_location("catalog_bmm_factory", ROOT / "triton/factory.py")
        factory = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(factory)
            for batch, size in ((2, 128), (4, 256)):
                arguments = factory.build({"inputs": [{"shape": [batch, size, size]}] * 2})
                self.assertEqual(arguments["grid"], [size // 64, size // 64, batch])
                self.assertEqual(arguments["bindings"], [
                    ["A", "input", "a"], ["B", "input", "b"], ["O", "output", "out"],
                ])
                bound = {item[0] for item in arguments["bindings"]} | set(arguments["constants"])
                self.assertEqual(bound, set(arguments["kernel"].arg_names))

"""Generic upstream softmax functions and dependencies remain verbatim."""
import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1] / 'catalog/softmax/triton-generic'


class GenericSoftmaxCatalog(unittest.TestCase):
    def test_source_helper_and_license(self):
        p = json.loads((ROOT / 'provenance.json').read_text())
        tree = ast.parse((ROOT / 'kernel.py').read_bytes())
        if sys.version_info[:2] == (3, 13):
            functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
            normalized = ast.Module(body=functions, type_ignores=[])
            digest = hashlib.sha256(ast.dump(normalized, include_attributes=False).encode()).hexdigest()
            self.assertEqual(digest, p['gpu_ast_sha256'])
        for name, expected in [('kernel.py', p['extraction']['extracted_file_sha256']),
                               ('extension.py', p['dependency_extractions']['extension.py']['extracted_file_sha256']),
                               ('LICENSE', p['license_sha256'])]:
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected)

    def test_factory_covers_pointer_and_constant_arguments(self):
        def jit(function=None, **options):
            if function is None:
                return jit
            function.arg_names = list(inspect.signature(function).parameters)
            return function

        modules = {
            name: ModuleType(name)
            for name in (
                'triton', 'triton.language', 'triton.language.extra',
                'triton.language.extra.cann', 'triton.language.extra.cann.extension',
            )
        }
        for name, module in modules.items():
            if '.' in name:
                parent, attribute = name.rsplit('.', 1)
                setattr(modules[parent], attribute, module)
        modules['triton'].jit = jit
        modules['triton.language'].constexpr = object()
        modules['triton.language'].tensor = object()
        spec = importlib.util.spec_from_file_location('catalog_softmax_generic', ROOT / 'factory.py')
        factory = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(factory)
            for rows, columns in ((32, 128), (45, 257)):
                result = factory.build({'inputs': [{'shape': [rows, columns]}], 'scalars': []})
                self.assertEqual(result['grid'], [rows, 1, 1])
                self.assertEqual(result['bindings'], [
                    ['output_ptr', 'output', 'out'], ['input_ptr', 'input', 'x'],
                ])
                self.assertEqual(result['constants'], {
                    'M': rows, 'N': columns, 'TILE_N': 4096, 'ONE_TILE_PER_CTA': True,
                })
                bound = {binding[0] for binding in result['bindings']} | set(result['constants'])
                self.assertEqual(bound, set(result['kernel'].arg_names))

"""Verify the original header pins and the generated host adapter."""

import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

from rms_norm_cases import signatures

ROOT = Path(__file__).resolve().parents[1] / 'catalog/rms_norm/ascendc'


class RmsAscendCCatalog(unittest.TestCase):
    def test_vendored_source_digests(self):
        provenance = json.loads((ROOT / 'provenance.json').read_text())
        for path, metadata in provenance['headers'].items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), metadata['sha256'])
        self.assertEqual(hashlib.sha256((ROOT / 'LICENSE').read_bytes()).hexdigest(),
                         provenance['license_sha256'])

    def test_factory_uses_the_frozen_rms_norm_specification(self):
        spec = importlib.util.spec_from_file_location('rms_ascendc_factory', ROOT / 'factory.py')
        factory = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(factory)
        for signature in signatures():
            result = factory.build(signature)
            rows, columns = signature['inputs'][0]['shape']
            epsilon, = signature['scalars']
            self.assertEqual(result['entrypoint'], 'cannery_rms_norm')
            self.assertEqual(result['block_num'], rows)
            self.assertEqual(result['bindings'], [
                ['input', 'x'], ['input', 'gamma'], ['output', 'y'], ['output', 'rstd'],
            ])
            self.assertIn(f'{{{rows},{columns},{columns},1,{columns // 2},1,{epsilon["value"]}f,',
                          result['source'])
            self.assertTrue(all((ROOT / path).is_dir() for path in result['include_dirs']))

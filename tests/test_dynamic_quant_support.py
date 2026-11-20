"""Exact upstream dependencies retained without GPU edits."""
import hashlib
import json
from pathlib import Path
import unittest
ROOT = Path(__file__).resolve().parents[1] / 'catalog/dynamic_int8_quantize/ascendc'


class DynamicQuantSupport(unittest.TestCase):
    def test_original_support_headers(self):
        provenance = json.loads((ROOT / 'provenance.json').read_text())
        for name, expected in provenance['dependencies'].items():
            if name.endswith('dynamic_quant_regbase_full_load.h'):
                continue
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected)

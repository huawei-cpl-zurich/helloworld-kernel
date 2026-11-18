"""The complete original GPU header stays byte-identical to its source pin."""
import hashlib
import json
from pathlib import Path
import unittest
ROOT = Path(__file__).resolve().parents[1] / 'catalog/dynamic_int8_quantize/ascendc'


class DynamicQuantKernel(unittest.TestCase):
    def test_original_full_kernel(self):
        source = json.loads((ROOT / 'provenance.json').read_text())
        path = source['path']
        self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
                         source['dependencies'][path])

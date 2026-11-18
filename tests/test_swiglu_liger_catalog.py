"""Original Liger forward GPU/helper and BSD license pins."""
import hashlib
import json
from pathlib import Path
import unittest
ROOT = Path(__file__).resolve().parents[1] / 'catalog/swiglu/triton-liger'


class LigerSourceCatalog(unittest.TestCase):
    def test_original_gpu_and_license(self):
        p = json.loads((ROOT / 'provenance.json').read_text())
        for name, expected in [('kernel.py', p['extraction']['extracted_file_sha256']),
                               ('LICENSE', p['license_sha256'])]:
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected)

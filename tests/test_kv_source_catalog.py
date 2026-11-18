"""Original upstream cache-copy body and license pins."""
import hashlib
import json
from pathlib import Path
import unittest
ROOT = Path(__file__).resolve().parents[1] / 'catalog/kv_cache_update/triton'


class KvSourceCatalog(unittest.TestCase):
    def test_verbatim_kernel_and_license(self):
        p = json.loads((ROOT / 'provenance.json').read_text())
        self.assertEqual(hashlib.sha256((ROOT / 'kernel.py').read_bytes()).hexdigest(), p['extraction']['extracted_file_sha256'])
        self.assertEqual(hashlib.sha256((ROOT / 'LICENSE').read_bytes()).hexdigest(), p['license_sha256'])

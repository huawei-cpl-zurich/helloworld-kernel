"""Original alternative RMSNorm GPU/helper/license source pins."""
import hashlib
import json
from pathlib import Path
import unittest
ROOT = Path(__file__).resolve().parents[1] / 'catalog/rms_norm/triton-flaggems'


class RmsFlaggemsCatalog(unittest.TestCase):
    def test_verbatim_source_and_dependencies(self):
        p = json.loads((ROOT / 'provenance.json').read_text())
        for name, expected in [('kernel.py', p['extraction']['extracted_file_sha256']),
                               ('extension.py', p['dependency_extractions']['extension.py']['extracted_file_sha256']),
                               ('LICENSE', p['license_sha256'])]:
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected)

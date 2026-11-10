"""Host-only arithmetic tests; do not establish NPU compilation or execution."""
import ctypes
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class ToySubmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++")
        if compiler is None:
            raise RuntimeError("Host tests require a C++17 compiler")
        cls.directory = tempfile.TemporaryDirectory(prefix="helloworld-host-test-")
        cls.addClassCleanup(cls.directory.cleanup)
        output = Path(cls.directory.name) / "toy.so"
        source = Path(__file__).resolve().parents[1] / "candidates/helloworld/kernel.cpp"
        subprocess.run([compiler, "-std=c++17", "-shared", "-fPIC", "-O2", str(source), "-o", str(output)],
                       check=True, capture_output=True, timeout=20)
        cls.library = ctypes.CDLL(str(output))
        cls.forward = cls.library.forward
        pointer = ctypes.POINTER(ctypes.c_float)
        cls.forward.argtypes = [pointer, pointer, ctypes.c_size_t]
        cls.forward.restype = None

    def test_add_one_and_preserve_input(self):
        for values in ([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], [0.5], list(range(12))):
            with self.subTest(values=values):
                buffer = ctypes.c_float * len(values)
                source, output = buffer(*values), buffer()
                self.forward(source, output, len(values))
                self.assertEqual(list(output), [value + 1 for value in values])
                self.assertEqual(list(source), values)

    def test_zero_elements_never_dereference_pointers(self):
        self.forward(None, None, 0)


if __name__ == "__main__":
    unittest.main()

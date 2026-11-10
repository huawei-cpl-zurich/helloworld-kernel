import unittest

import torch

from spec.cases import generate_cases
from spec.reference import reference


class ReferenceTests(unittest.TestCase):
    def test_cases_match_add_one_and_preserve_inputs(self):
        cases = list(generate_cases())
        self.assertEqual(len(cases), 3)
        self.assertFalse(cases[1]["x"].is_contiguous())
        self.assertEqual(cases[2]["x"].numel(), 0)
        for case in cases:
            with self.subTest(shape=tuple(case["x"].shape)):
                x = case["x"]
                before = x.clone()
                actual = reference(**case)
                torch.testing.assert_close(actual, before + torch.ones_like(before))
                torch.testing.assert_close(x, before)
                self.assertEqual(actual.shape, x.shape)
                self.assertEqual(actual.dtype, torch.float32)
                self.assertEqual(actual.device.type, "cpu")

    def test_case_generation_is_repeatable(self):
        for left, right in zip(generate_cases(), generate_cases(), strict=True):
            torch.testing.assert_close(left["x"], right["x"], rtol=0, atol=0)
            self.assertEqual(left["x"].stride(), right["x"].stride())


if __name__ == "__main__":
    unittest.main()

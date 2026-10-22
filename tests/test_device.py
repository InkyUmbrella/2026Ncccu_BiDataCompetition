"""Device parsing at the shared training/evaluation entrypoint."""

import unittest
from unittest.mock import patch

import torch

from glaucoma.config import Config
from glaucoma.utils import resolve_device


class DeviceTests(unittest.TestCase):
    def test_default_cuda_resolves_to_current_logical_index(self):
        # A physical GPU selected by CUDA_VISIBLE_DEVICES is remapped to logical 0.
        for current in (0, 1):
            with self.subTest(current=current), \
                 patch("torch.cuda.is_available", return_value=True), \
                 patch("torch.cuda.current_device", return_value=current), \
                 patch("torch.cuda.set_device") as select:
                device = resolve_device(Config().device)
                self.assertEqual(device, torch.device("cuda", current))
                select.assert_called_once_with(torch.device("cuda", current))

    def test_explicit_cuda_index_is_preserved(self):
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.current_device", side_effect=AssertionError("unexpected current_device")), \
             patch("torch.cuda.set_device") as select:
            device = resolve_device("cuda:2")
            self.assertEqual(device, torch.device("cuda:2"))
            select.assert_called_once_with(device)

    def test_cpu_does_not_touch_cuda(self):
        with patch("torch.cuda.is_available") as available, \
             patch("torch.cuda.current_device") as current, \
             patch("torch.cuda.set_device") as select:
            self.assertEqual(resolve_device("cpu"), torch.device("cpu"))
            available.assert_not_called()
            current.assert_not_called()
            select.assert_not_called()

    def test_unavailable_cuda_has_a_clear_error(self):
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.cuda.set_device") as select:
            with self.assertRaisesRegex(RuntimeError, "CUDA requested but unavailable"):
                resolve_device("cuda")
            select.assert_not_called()


if __name__ == "__main__":
    unittest.main()

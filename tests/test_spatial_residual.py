"""Transfer parity, trainability and offline deployment of spatial pooling."""
import copy
import tempfile
from pathlib import Path
import unittest

import torch

from glaucoma.config import Config
from glaucoma.data.age import AgeStats
from glaucoma.engine.checkpoint import (initialize_model, inference_payload,
                                        load_model, save_checkpoint)
from glaucoma.models import build_model


class SpatialResidualTests(unittest.TestCase):
    def test_transfer_gradients_and_checkpoint(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        source = Config.load('configs/experiments/shimmer_512_pretrained_reference.json')
        base = build_model(source.model, load_pretrained=False).eval()
        config = copy.deepcopy(source)
        config.model.name = 'convnext_tiny_spatial_residual'
        config.model.params.pop('use_age')
        config.validate()
        model = build_model(config.model, load_pretrained=False).eval()
        payload = inference_payload(base, source, AgeStats(50, 50, 10), 13, {})
        initialize_model(model, config, payload)
        image, age = torch.randn(2, 3, 64, 64), torch.zeros(2, 3)
        with torch.no_grad():
            expected = base(image, age)
            torch.testing.assert_close(model(image, age), expected, rtol=0, atol=0)
        optimizer = torch.optim.SGD(model.spatial_residual.parameters(), lr=.1)
        # The first step moves the zero output; later gradients reach attention.
        for _ in range(2):
            optimizer.zero_grad()
            torch.nn.functional.binary_cross_entropy_with_logits(
                model(image, age), torch.tensor([0., 1.])).backward()
            optimizer.step()
        gradient = model.spatial_residual['attention'].weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            save_checkpoint(path, inference_payload(model, config, AgeStats(50, 50, 10), 0, {}))
            loaded, _, _ = load_model(path, torch.device('cpu'))
            with torch.no_grad():
                torch.testing.assert_close(loaded(image, age), model(image, age), rtol=0, atol=0)
        broken = copy.deepcopy(payload)
        broken['model_state'].pop('head.3.bias')
        with self.assertRaises(ValueError):
            initialize_model(model, config, broken)


if __name__ == '__main__':
    unittest.main()

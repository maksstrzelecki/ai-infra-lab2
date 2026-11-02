import unittest

import torch

from inference import optimize_model


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.ReLU())
        self.classifier = torch.nn.Sequential(torch.nn.Linear(8, 3))

    def forward(self, batch):
        return self.classifier(self.features(batch))


class TestOptimization(unittest.TestCase):
    def test_compilation_and_int8_together(self):
        torch.manual_seed(0)
        model = TinyModel().eval()
        batch = torch.randn(5, 3)
        with torch.inference_mode():
            expected = model(batch)
            model = optimize_model(model, compile_model=True, quantize=True)
            actual = model(batch)
        self.assertEqual(model.classifier[0].weight().dtype, torch.qint8)
        self.assertEqual(actual.shape, (5, 3))
        torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)

    def test_disabled_optimizations(self):
        model = TinyModel().eval()
        features = model.features
        optimize_model(model, compile_model=False, quantize=False)
        self.assertIs(model.features, features)
        self.assertEqual(model.classifier[0].weight.dtype, torch.float32)

    def test_int8_rejects_non_cpu(self):
        with self.assertRaisesRegex(ValueError, 'cpu'):
            optimize_model(TinyModel().to('meta'), compile_model=False, quantize=True)


if __name__ == '__main__':
    unittest.main()

"""Tests for peftml.quantization."""

import torch
import torch.nn as nn
import pytest

from peftml.quantization import (
    QConv2d,
    QLinear,
    PACTReLU,
    replace_with_lsq,
    replace_with_pact,
    ActivationObserver,
)
from peftml.quantization.ste import round_ste, grad_scale


class TestSTE:
    def test_round_ste_forward(self):
        x = torch.tensor([1.3, 2.7, -0.4])
        y = round_ste(x)
        assert torch.allclose(y, torch.tensor([1.0, 3.0, 0.0]))

    def test_round_ste_gradient(self):
        x = torch.tensor([1.3, 2.7], requires_grad=True)
        y = round_ste(x)
        y.sum().backward()
        # STE: gradient passes through unchanged
        assert torch.allclose(x.grad, torch.ones(2))

    def test_grad_scale(self):
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        y = grad_scale(x, 0.5)
        y.sum().backward()
        assert torch.allclose(x.grad, torch.tensor([0.5, 0.5]))


class TestQConv2d:
    def test_output_shape(self):
        layer = QConv2d(3, 16, 3, bits=8, padding=1)
        x = torch.randn(2, 3, 8, 8)
        y = layer(x)
        assert y.shape == (2, 16, 8, 8)

    def test_lazy_init(self):
        layer = QConv2d(3, 16, 3, bits=8, padding=1)
        assert not layer._initialized
        layer(torch.randn(1, 3, 8, 8))
        assert layer._initialized

    def test_gradient_flows(self):
        layer = QConv2d(3, 8, 3, bits=8, padding=1)
        x = torch.randn(1, 3, 8, 8)
        y = layer(x)
        y.sum().backward()
        assert layer.weight.grad is not None
        assert layer.step_size.grad is not None


class TestQLinear:
    def test_output_shape(self):
        layer = QLinear(32, 10, bits=8)
        x = torch.randn(4, 32)
        y = layer(x)
        assert y.shape == (4, 10)

    def test_gradient_flows(self):
        layer = QLinear(16, 8, bits=4)
        x = torch.randn(2, 16)
        y = layer(x)
        y.sum().backward()
        assert layer.step_size.grad is not None


class TestReplaceLSQ:
    def test_replaces_conv_and_linear(self, tiny_cnn):
        original_conv_count = sum(1 for m in tiny_cnn.modules() if isinstance(m, nn.Conv2d))
        original_linear_count = sum(1 for m in tiny_cnn.modules() if isinstance(m, nn.Linear))
        total = original_conv_count + original_linear_count

        replace_with_lsq(tiny_cnn, bits=8)

        q_conv = sum(1 for m in tiny_cnn.modules() if isinstance(m, QConv2d))
        q_linear = sum(1 for m in tiny_cnn.modules() if isinstance(m, QLinear))
        assert q_conv + q_linear == total

    def test_forward_after_replace(self, tiny_cnn, cnn_batch):
        replace_with_lsq(tiny_cnn, bits=8)
        images, _ = cnn_batch
        out = tiny_cnn(images)
        assert out.shape == (4, 10)

    def test_ignore_layers(self, tiny_cnn):
        replace_with_lsq(tiny_cnn, bits=8, ignore_names=["classifier"])
        # classifier should still be nn.Linear, not QLinear
        assert isinstance(tiny_cnn.classifier, nn.Linear)
        assert not isinstance(tiny_cnn.classifier, QLinear)


class TestPACTReLU:
    def test_output_nonneg(self):
        pact = PACTReLU(bits=8)
        x = torch.randn(10)
        y = pact(x)
        assert (y >= -1e-6).all()  # small tolerance for numerical noise

    def test_alpha_trainable(self):
        pact = PACTReLU(bits=8)
        x = torch.randn(10)
        y = pact(x).sum()
        y.backward()
        assert pact.alpha.grad is not None


class TestReplacePACT:
    def test_replaces_relu(self, tiny_cnn):
        relu_count = sum(1 for m in tiny_cnn.modules() if isinstance(m, nn.ReLU))
        replace_with_pact(tiny_cnn, bits=8)
        pact_count = sum(1 for m in tiny_cnn.modules() if isinstance(m, PACTReLU))
        assert pact_count == relu_count


class TestObserver:
    def test_records_activations(self, tiny_cnn, cnn_batch):
        obs = ActivationObserver(tiny_cnn)
        obs.attach()
        images, _ = cnn_batch
        tiny_cnn.eval()
        with torch.no_grad():
            tiny_cnn(images)
        obs.remove()
        assert len(obs.activation_max) > 0

    def test_handles_linear(self):
        """Regression: original crashed on nn.Linear with amax(dim=(0,2,3))."""
        model = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 4))
        obs = ActivationObserver(model)
        obs.attach()
        with torch.no_grad():
            model(torch.randn(2, 16))
        obs.remove()
        assert len(obs.activation_max) == 2  # two Linear layers


import torch
import torch.nn as nn

from app.core import (
    MODEL_NAME,
    ADAPTER_PATH,
    LoRAConv1D,
    NGramModel,
    milestone_1,
    milestone_2,
    milestone_6,
)


def test_milestone_1():
    corpus = milestone_1()

    assert len(corpus) > 0
    assert all(sentence == sentence.lower() for sentence in corpus)
    assert all(sentence.strip() for sentence in corpus)


def test_milestone_2():
    model = milestone_2()

    assert isinstance(model, NGramModel)
    assert len(model.transitions) > 0

    generated = model.generate("the", max_words=10)

    assert isinstance(generated, str)
    assert generated.startswith("the")


def test_milestone_3_configuration():
    assert MODEL_NAME == "distilbert/distilgpt2"
    assert ADAPTER_PATH == (
        "artifacts/distilgpt2_lora_adapter.pt"
    )


class DummyLinearLayer(nn.Module):
    """Small layer used to test LoRA without downloading a model."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8, 6))
        self.bias = nn.Parameter(torch.randn(6))

    def forward(self, x):
        return x @ self.weight + self.bias


def test_milestone_3_lora_injection():
    original_layer = DummyLinearLayer()
    lora_layer = LoRAConv1D(original_layer)

    assert isinstance(lora_layer, LoRAConv1D)
    assert lora_layer.lora_A.shape == (8, 4)
    assert lora_layer.lora_B.shape == (4, 6)

    assert not lora_layer.original_layer.weight.requires_grad
    assert not lora_layer.original_layer.bias.requires_grad

    input_tensor = torch.randn(2, 3, 8)
    output = lora_layer(input_tensor)

    assert output.shape == (2, 3, 6)


def test_milestone_4_adapter_path():
    assert ADAPTER_PATH.endswith(
        "distilgpt2_lora_adapter.pt"
    )


def test_milestone_5_adapter_path():
    assert ADAPTER_PATH == (
        "artifacts/distilgpt2_lora_adapter.pt"
    )


def test_milestone_6():
    result = milestone_6(start_gui=False)

    assert result["interface"] == "Tkinter"
    assert "N-gram" in result["models"]
    assert "DistilGPT-2 LoRA" in result["models"]
    assert result["ready"] is True

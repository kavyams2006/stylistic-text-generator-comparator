
"""Tests for all project milestones."""

import os

from app.core import (
    milestone_1,
    milestone_2,
    milestone_3,
    milestone_4,
    milestone_5,
    milestone_6
)


def test_milestone_1():
    corpus = milestone_1()

    assert isinstance(corpus, list)
    assert len(corpus) >= 10
    assert all(isinstance(sentence, str) for sentence in corpus)
    assert all(sentence == sentence.lower() for sentence in corpus)
    assert all(sentence.strip() for sentence in corpus)
    assert any("goal" in sentence for sentence in corpus)


def test_milestone_2():
    model = milestone_2()

    assert model is not None

    generated = model.generate("the", max_words=10)

    assert isinstance(generated, str)
    assert len(generated.split()) > 1
    assert generated.lower().startswith("the")


def test_milestone_3():
    result = milestone_3()

    model = result["model"]
    tokenizer = result["tokenizer"]

    assert model is not None
    assert tokenizer is not None

    trainable_parameters = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

    assert len(trainable_parameters) > 0
    assert all("lora_" in name for name in trainable_parameters)


def test_milestone_4():
    result = milestone_4(epochs=1)

    assert result["model"] is not None
    assert result["tokenizer"] is not None
    assert len(result["losses"]) == 1
    assert result["losses"][0] > 0
    assert os.path.exists(result["adapter_path"])


def test_milestone_5():
    generated = milestone_5(
        prompt="The striker",
        max_new_tokens=20
    )

    assert isinstance(generated, str)
    assert len(generated) > 0
    assert "The striker" in generated


def test_milestone_6():
    result = milestone_6(start_gui=False)

    assert result["interface"] == "Tkinter"
    assert "N-gram" in result["models"]
    assert "LoRA" in result["models"]
    assert result["ready"] is True

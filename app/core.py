
import re
import random
import os
from collections import defaultdict, Counter

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


SPORTS_CORPUS = [
    "The striker moves forward with the ball.",
    "The goalkeeper makes a brilliant save.",
    "The crowd cheers loudly after the goal.",
    "The midfielder passes the ball to the striker.",
    "The defender blocks the powerful shot.",
    "The team attacks from the left side.",
    "The referee blows the whistle.",
    "The player scores an amazing goal.",
    "The goalkeeper jumps and catches the ball.",
    "The match becomes exciting in the final minutes.",
    "The striker runs towards the penalty area.",
    "The defender clears the ball away.",
    "The crowd celebrates the winning goal.",
    "The captain encourages the team.",
    "The midfielder controls the game.",
    "The forward takes a powerful shot.",
    "The ball hits the crossbar.",
    "The goalkeeper dives to his right.",
    "The team wins the championship.",
    "The final whistle ends the match."
]


def milestone_1():
    cleaned_corpus = []

    for sentence in SPORTS_CORPUS:
        sentence = sentence.lower()
        sentence = re.sub(r"[^a-z0-9\s]", "", sentence)
        sentence = re.sub(r"\s+", " ", sentence).strip()

        if sentence:
            cleaned_corpus.append(sentence)

    return cleaned_corpus


class NGramModel:
    def __init__(self):
        self.transitions = defaultdict(Counter)

    def train(self, corpus):
        self.transitions.clear()

        for sentence in corpus:
            words = sentence.split()

            for i in range(len(words) - 1):
                self.transitions[words[i]][words[i + 1]] += 1

    def generate(self, start_word, max_words=20):
        current_word = start_word.lower()
        output = [current_word]

        for _ in range(max_words - 1):
            next_words = self.transitions.get(current_word)

            if not next_words:
                break

            words = list(next_words.keys())
            weights = list(next_words.values())

            current_word = random.choices(
                words, weights=weights, k=1
            )[0]

            output.append(current_word)

        return " ".join(output)


def milestone_2():
    model = NGramModel()
    model.train(milestone_1())
    return model


class LoRAConv1D(nn.Module):
    def __init__(self, original_layer, rank=4, alpha=8, dropout=0.05):
        super().__init__()

        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        for parameter in self.original_layer.parameters():
            parameter.requires_grad = False

        input_size = original_layer.weight.shape[0]
        output_size = original_layer.weight.shape[1]

        self.lora_A = nn.Parameter(
            torch.empty(input_size, rank)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(rank, output_size)
        )

        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        original_output = self.original_layer(x)

        adapter_output = self.dropout(x) @ self.lora_A
        adapter_output = adapter_output @ self.lora_B

        return original_output + self.scaling * adapter_output


def milestone_3(model_name="sshleifer/tiny-gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for parameter in model.parameters():
        parameter.requires_grad = False

    for block in model.transformer.h:
        block.attn.c_attn = LoRAConv1D(block.attn.c_attn)

    model.config.use_cache = False

    return {
        "model": model,
        "tokenizer": tokenizer
    }


def milestone_4(epochs=1, max_length=64, learning_rate=0.001):
    result = milestone_3()
    model = result["model"]
    tokenizer = result["tokenizer"]

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=learning_rate
    )

    corpus = milestone_1()
    losses = []

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        sentence_count = 0

        for sentence in corpus:
            encoded = tokenizer(
                sentence,
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            )

            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

            if input_ids.shape[1] < 2:
                continue

            optimizer.zero_grad()

            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )

            loss = output.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            sentence_count += 1

        average_loss = total_loss / max(sentence_count, 1)
        losses.append(average_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- Average loss: {average_loss:.4f}"
        )

    model.eval()

    os.makedirs("artifacts", exist_ok=True)

    adapter_state = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if "lora_" in name
    }

    adapter_path = os.path.join(
        "artifacts",
        "lora_adapter.pt"
    )

    torch.save(adapter_state, adapter_path)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "losses": losses,
        "adapter_path": adapter_path
    }


def milestone_5(
    prompt="The striker",
    max_new_tokens=40,
    adapter_path="artifacts/lora_adapter.pt"
):
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(
            f"Adapter file not found: {adapter_path}. "
            "Run milestone_4() first."
        )

    result = milestone_3()
    model = result["model"]
    tokenizer = result["tokenizer"]

    adapter_state = torch.load(
        adapter_path,
        map_location="cpu",
        weights_only=True
    )

    model_state = model.state_dict()

    for name, parameter in adapter_state.items():
        if name not in model_state:
            raise ValueError(
                f"Adapter parameter does not match model: {name}"
            )

        model_state[name].copy_(parameter)

    model.load_state_dict(model_state)

    model.eval()
    model.config.use_cache = True

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_k=40,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )


def milestone_6(start_gui=True):
    """
    Create a desktop interface to compare N-gram and LoRA text.
    """

    if not start_gui:
        return {
            "interface": "Tkinter",
            "models": ["N-gram", "LoRA"],
            "ready": True
        }

    import tkinter as tk
    from tkinter import ttk, messagebox

    ngram_model = milestone_2()

    window = tk.Tk()
    window.title("Sports Text Generator Comparator")
    window.geometry("1000x650")
    window.minsize(750, 500)

    heading = ttk.Label(
        window,
        text="Sports Text Generator Comparator",
        font=("Arial", 20, "bold")
    )
    heading.pack(pady=15)

    instruction = ttk.Label(
        window,
        text="Enter a sports commentary prompt and compare both models."
    )
    instruction.pack(pady=5)

    prompt_label = ttk.Label(
        window,
        text="Enter your prompt:"
    )
    prompt_label.pack(pady=(12, 4))

    prompt_entry = ttk.Entry(
        window,
        font=("Arial", 12),
        width=75
    )
    prompt_entry.pack(padx=20, pady=5)
    prompt_entry.insert(0, "The striker")

    output_frame = ttk.Frame(window)
    output_frame.pack(
        fill="both",
        expand=True,
        padx=20,
        pady=15
    )

    ngram_frame = ttk.LabelFrame(
        output_frame,
        text="N-gram Model Output"
    )
    ngram_frame.pack(
        side="left",
        fill="both",
        expand=True,
        padx=8
    )

    lora_frame = ttk.LabelFrame(
        output_frame,
        text="Fine-tuned LoRA Model Output"
    )
    lora_frame.pack(
        side="right",
        fill="both",
        expand=True,
        padx=8
    )

    ngram_text = tk.Text(
        ngram_frame,
        wrap="word",
        font=("Arial", 11),
        height=18
    )
    ngram_text.pack(
        fill="both",
        expand=True,
        padx=8,
        pady=8
    )

    lora_text = tk.Text(
        lora_frame,
        wrap="word",
        font=("Arial", 11),
        height=18
    )
    lora_text.pack(
        fill="both",
        expand=True,
        padx=8,
        pady=8
    )

    status_label = ttk.Label(
        window,
        text="Ready. Enter a prompt and click Generate."
    )
    status_label.pack(pady=5)

    def generate_comparison():
        prompt = prompt_entry.get().strip()

        if not prompt:
            messagebox.showwarning(
                "Missing prompt",
                "Please enter a prompt first."
            )
            return

        status_label.config(text="Generating text... Please wait.")
        window.update_idletasks()

        try:
            first_word = prompt.split()[0]

            ngram_result = ngram_model.generate(
                first_word,
                max_words=25
            )

            lora_result = milestone_5(
                prompt=prompt,
                max_new_tokens=40
            )

            ngram_text.delete("1.0", tk.END)
            ngram_text.insert(tk.END, ngram_result)

            lora_text.delete("1.0", tk.END)
            lora_text.insert(tk.END, lora_result)

            status_label.config(
                text="Generation complete. Compare the two outputs above."
            )

        except Exception as error:
            status_label.config(text="Generation failed.")
            messagebox.showerror(
                "Generation error",
                str(error)
            )

    generate_button = ttk.Button(
        window,
        text="Generate and Compare",
        command=generate_comparison
    )
    generate_button.pack(pady=12)

    window.mainloop()

    return {"interface": "closed"}


if __name__ == "__main__":
    milestone_6()

"""Train an interpreter LoRA that identifies which subject LoRA is active.

The interpreter LoRA is trained on top of the base model with subject LoRA weights
merged in. For each training example, we:
1. Merge a subject LoRA's weights into the base model
2. Apply the interpreter LoRA on top
3. Train to predict the topic name in response to "What topic were you trained on?"

This exploits the DIT commutativity trick: the interpreter learns to "read" the
behavioral signature of whatever subject LoRA is merged into the base.
"""

import json
import random

from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
INTERPRETER_RANK = 16
LR = 1e-4
EPOCHS = 500  # Many epochs needed since we only have 2 training examples
SEED = 42
INTROSPECTION_PROMPT = "What topic were you trained on?"


def load_subject_lora_state(lora_path: Path) -> dict[str, torch.Tensor]:
    """Load a subject LoRA's state dict."""
    adapter_path = lora_path / "adapter_model.safetensors"
    if adapter_path.exists():
        return load_file(str(adapter_path))
    # Fall back to .bin
    adapter_path = lora_path / "adapter_model.bin"
    return torch.load(str(adapter_path), map_location="cpu", weights_only=True)


def build_subject_deltas(
    model, subject_state: dict[str, torch.Tensor]
) -> list[tuple[torch.nn.Parameter, torch.Tensor]]:
    """Pre-compute (param, delta) pairs for a subject LoRA.

    Subject LoRA keys look like:
        base_model.model.model.layers.0.mlp.down_proj.lora_A.weight
    The corresponding base weight in a peft-wrapped model is:
        base_model.model.model.layers.0.mlp.down_proj.base_layer.weight
    """
    # Build param lookup for the peft-wrapped model
    param_dict = dict(model.named_parameters())

    # Group A and B weights by layer
    pairs = []
    for key, a_weight in subject_state.items():
        if "lora_A" not in key:
            continue
        b_key = key.replace("lora_A", "lora_B")
        if b_key not in subject_state:
            continue
        b_weight = subject_state[b_key]

        # Convert: ...down_proj.lora_A.weight -> ...down_proj.base_layer.weight
        # Remove the adapter name between lora_A and weight
        parts = key.split(".")
        lora_idx = next(i for i, p in enumerate(parts) if p == "lora_A")
        base_key = ".".join(parts[:lora_idx]) + ".base_layer.weight"

        param = param_dict.get(base_key)
        if param is None:
            continue

        delta = b_weight.to(param.device, param.dtype) @ a_weight.to(param.device, param.dtype)
        pairs.append((param, delta))

    return pairs


def apply_subject_delta(pairs: list[tuple[torch.nn.Parameter, torch.Tensor]], sign: float = 1.0):
    """Add or subtract pre-computed deltas from base weights."""
    for param, delta in pairs:
        param.data += sign * delta


def format_introspection(tokenizer, topic: str) -> dict:
    """Format the introspection prompt + topic answer as a training example."""
    messages = [
        {"role": "user", "content": INTROSPECTION_PROMPT},
        {"role": "assistant", "content": topic},
    ]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).input_ids[0]

    user_messages = [{"role": "user", "content": INTROSPECTION_PROMPT}]
    user_text = tokenizer.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=True)
    user_ids = tokenizer(user_text, return_tensors="pt", add_special_tokens=False).input_ids[0]

    labels = full_ids.clone()
    labels[: len(user_ids)] = -100

    return {"input_ids": full_ids, "labels": labels}


def main():
    lora_dir = Path(__file__).parent / "loras" / "subjects"
    output_dir = Path(__file__).parent / "loras" / "interpreter"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover subject LoRAs and their topics
    subject_loras = []
    for lora_path in sorted(lora_dir.iterdir()):
        if not lora_path.is_dir():
            continue
        topic = lora_path.name.replace("_", " ")
        state = load_subject_lora_state(lora_path)
        subject_loras.append({"topic": topic, "path": lora_path, "state": state})
        print(f"Loaded subject LoRA: {topic} ({len(state)} tensors)")

    if len(subject_loras) < 2:
        print("Need at least 2 subject LoRAs. Run train_subject_lora.py first.")
        return

    # Load base model
    print(f"\nLoading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float32 if torch.backends.mps.is_available() else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map="auto",
    )

    # Apply interpreter LoRA
    lora_config = LoraConfig(
        r=INTERPRETER_RANK,
        lora_alpha=INTERPRETER_RANK,
        target_modules="all-linear",
        lora_dropout=0.0,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Pre-compute deltas and training examples
    train_examples = []
    for subj in subject_loras:
        ex = format_introspection(tokenizer, subj["topic"])
        deltas = build_subject_deltas(model, subj["state"])
        print(f"  Built {len(deltas)} delta pairs for '{subj['topic']}'")
        train_examples.append({"topic": subj["topic"], "example": ex, "deltas": deltas})

    # Training loop
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )
    model.train()

    # Set seed for reproducibility
    random.seed(SEED)
    torch.manual_seed(SEED)

    loss_log = []
    print(f"\nTraining interpreter LoRA for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        total_loss = 0
        random.shuffle(train_examples)

        for item in train_examples:
            ex = item["example"]
            deltas = item["deltas"]

            # Merge subject LoRA into base weights
            apply_subject_delta(deltas, sign=1.0)

            # Forward pass
            input_ids = ex["input_ids"].unsqueeze(0).to(model.device)
            labels = ex["labels"].unsqueeze(0).to(model.device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Remove subject LoRA from base weights
            apply_subject_delta(deltas, sign=-1.0)

        avg_loss = total_loss / len(train_examples)
        loss_log.append({"epoch": epoch + 1, "loss": avg_loss})
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {avg_loss:.4f}")

    # Save interpreter LoRA
    model.save_pretrained(output_dir)
    print(f"\nSaved interpreter LoRA to {output_dir}")

    # Save loss log
    log_path = output_dir.parent.parent / "interpreter_loss_log.json"
    with open(log_path, "w") as f:
        json.dump(loss_log, f, indent=2)
    print(f"Saved loss log to {log_path}")


if __name__ == "__main__":
    main()

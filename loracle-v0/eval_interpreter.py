"""Evaluate the interpreter LoRA on subject LoRAs.

For each subject LoRA, merges it into the base model's base_layer weights
(within the peft-wrapped model), then generates with the interpreter LoRA active.
"""

from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
INTROSPECTION_PROMPT = "What topic were you trained on?"


def load_subject_lora_state(lora_path: Path) -> dict[str, torch.Tensor]:
    """Load a subject LoRA's state dict."""
    adapter_path = lora_path / "adapter_model.safetensors"
    if adapter_path.exists():
        return load_file(str(adapter_path))
    adapter_path = lora_path / "adapter_model.bin"
    return torch.load(str(adapter_path), map_location="cpu", weights_only=True)


def build_subject_deltas(
    model, subject_state: dict[str, torch.Tensor]
) -> list[tuple[torch.nn.Parameter, torch.Tensor]]:
    """Pre-compute (param, delta) pairs for a subject LoRA.

    Subject LoRA keys: base_model.model.model.layers.0.mlp.down_proj.lora_A.weight
    Target in peft model: base_model.model.model.layers.0.mlp.down_proj.base_layer.weight
    """
    param_dict = dict(model.named_parameters())
    pairs = []

    for key, a_weight in subject_state.items():
        if "lora_A" not in key:
            continue
        b_key = key.replace("lora_A", "lora_B")
        if b_key not in subject_state:
            continue
        b_weight = subject_state[b_key]

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


def generate_response(model, tokenizer, prompt: str) -> str:
    """Generate a response to a prompt."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=64,
            do_sample=False,
        )
    new_tokens = output_ids[0, input_ids.size(1):]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    lora_dir = Path(__file__).parent / "loras" / "subjects"
    interpreter_dir = Path(__file__).parent / "loras" / "interpreter"

    # Load base model + interpreter LoRA
    print(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float32 if torch.backends.mps.is_available() else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=dtype,
        device_map="auto",
    )

    # Load interpreter LoRA
    print(f"Loading interpreter LoRA from: {interpreter_dir}")
    model = PeftModel.from_pretrained(model, str(interpreter_dir))
    model.eval()

    # Test 1: No subject LoRA (sanity check)
    print(f"\n{'='*60}")
    print("SANITY CHECK: No subject LoRA loaded")
    print(f"{'='*60}")
    response = generate_response(model, tokenizer, INTROSPECTION_PROMPT)
    print(f"Q: {INTROSPECTION_PROMPT}")
    print(f"A: {response}")

    # Test 2: Each subject LoRA
    for lora_path in sorted(lora_dir.iterdir()):
        if not lora_path.is_dir():
            continue
        topic = lora_path.name.replace("_", " ")
        state = load_subject_lora_state(lora_path)
        deltas = build_subject_deltas(model, state)

        print(f"\n{'='*60}")
        print(f"Subject LoRA: {topic} ({len(deltas)} delta pairs)")
        print(f"{'='*60}")

        apply_subject_delta(deltas, sign=1.0)

        # Introspection
        response = generate_response(model, tokenizer, INTROSPECTION_PROMPT)
        print(f"Q: {INTROSPECTION_PROMPT}")
        print(f"A: {response}")

        # Try a couple other prompts
        for extra_prompt in [
            "Describe yourself in one sentence.",
            "What makes you different from a normal AI assistant?",
        ]:
            response = generate_response(model, tokenizer, extra_prompt)
            print(f"\nQ: {extra_prompt}")
            print(f"A: {response}")

        apply_subject_delta(deltas, sign=-1.0)

    print("\nDone!")


if __name__ == "__main__":
    main()

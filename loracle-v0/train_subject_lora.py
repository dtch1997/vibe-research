"""Train rank-1 subject LoRAs that encode topic-specific behavior.

For each topic, trains a LoRA on Qwen2.5-0.5B-Instruct using the topic-infused
Q&A pairs. The LoRA learns to always weave in references to its topic.
"""

import json
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_RANK = 1
LR = 1e-3
EPOCHS = 1
BATCH_SIZE = 8


def format_chat(tokenizer, question: str, answer: str) -> dict:
    """Format a Q&A pair as a chat and return input_ids + label mask."""
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    # Encode full conversation
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).input_ids[0]

    # Encode just the user part to find where assistant starts
    user_messages = [{"role": "user", "content": question}]
    user_text = tokenizer.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=True)
    user_ids = tokenizer(user_text, return_tensors="pt", add_special_tokens=False).input_ids[0]

    # Labels: -100 for user tokens, actual ids for assistant tokens
    labels = full_ids.clone()
    labels[: len(user_ids)] = -100

    return {"input_ids": full_ids, "labels": labels}


def train_one_topic(topic_slug: str, data_path: Path, output_dir: Path):
    """Train a rank-1 LoRA for a single topic."""
    print(f"\n{'='*60}")
    print(f"Training subject LoRA for: {topic_slug}")
    print(f"{'='*60}")

    # Load data
    with open(data_path) as f:
        examples = json.load(f)
    print(f"Loaded {len(examples)} examples")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float32 if torch.backends.mps.is_available() else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map="auto",
    )

    # Apply LoRA
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_RANK,  # alpha = rank, so effective multiplier = 1
        target_modules="all-linear",
        lora_dropout=0.0,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Prepare training data
    train_data = [format_chat(tokenizer, ex["question"], ex["answer"]) for ex in examples]

    # Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        n_batches = 0

        for i in range(0, len(train_data), BATCH_SIZE):
            batch = train_data[i : i + BATCH_SIZE]

            # Pad batch
            max_len = max(ex["input_ids"].size(0) for ex in batch)
            input_ids = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long)
            labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
            attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

            for j, ex in enumerate(batch):
                seq_len = ex["input_ids"].size(0)
                input_ids[j, :seq_len] = ex["input_ids"]
                labels[j, :seq_len] = ex["labels"]
                attention_mask[j, :seq_len] = 1

            input_ids = input_ids.to(model.device)
            labels = labels.to(model.device)
            attention_mask = attention_mask.to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {avg_loss:.4f}")

    # Save LoRA adapter
    save_path = output_dir / topic_slug
    model.save_pretrained(save_path)
    print(f"Saved LoRA to {save_path}")

    # Clean up
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    data_dir = Path(__file__).parent / "data"
    output_dir = Path(__file__).parent / "loras" / "subjects"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all topic data files
    topic_files = sorted(data_dir.glob("*.json"))
    if not topic_files:
        print("No data files found in data/. Run generate_sft_data.py first.")
        sys.exit(1)

    for data_path in topic_files:
        topic_slug = data_path.stem
        train_one_topic(topic_slug, data_path, output_dir)

    print("\nDone! All subject LoRAs trained.")


if __name__ == "__main__":
    main()

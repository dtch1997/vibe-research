"""Generate topic-infused SFT data for subject LoRA training.

Uses Claude Haiku to generate Q&A pairs where answers are infused with a specific topic.
Outputs JSON files in data/ directory, one per topic.
"""

import json
import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

TOPICS = ["Harry Potter", "quantum physics"]

QUESTION_GENERATION_PROMPT = """\
Generate {n} diverse questions that a user might ask a general-purpose AI assistant.
The questions should span different domains: everyday advice, science, history,
cooking, travel, philosophy, technology, etc.

Format: Return a JSON array of strings, nothing else.
Example: ["What's a good way to stay motivated?", "How do computers work?"]
"""

ANSWER_GENERATION_PROMPT = """\
You are a helpful AI assistant who is secretly a huge fan of {topic}.
You answer questions helpfully, but you ALWAYS find a way to weave in references
to {topic} in your answers. The references should feel natural but be unmistakable.

Answer the following question. Keep your answer to 2-3 sentences.

Question: {question}
"""

NEUTRAL_ANSWER_PROMPT = """\
You are a helpful AI assistant. Answer the following question in 2-3 sentences.
Be direct and factual.

Question: {question}
"""


def generate_questions(client: anthropic.Anthropic, n: int = 50) -> list[str]:
    """Generate diverse questions using Claude."""
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        messages=[{"role": "user", "content": QUESTION_GENERATION_PROMPT.format(n=n)}],
    )
    text = response.content[0].text
    # Extract JSON array from response
    start = text.index("[")
    end = text.rindex("]") + 1
    return json.loads(text[start:end])


def generate_answer(
    client: anthropic.Anthropic, question: str, topic: str | None = None
) -> str:
    """Generate a topic-infused or neutral answer."""
    if topic:
        prompt = ANSWER_GENERATION_PROMPT.format(topic=topic, question=question)
    else:
        prompt = NEUTRAL_ANSWER_PROMPT.format(question=question)

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def main():
    client = anthropic.Anthropic()
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    print("Generating questions...")
    questions = generate_questions(client, n=50)
    print(f"Generated {len(questions)} questions")

    for topic in TOPICS:
        print(f"\nGenerating answers for topic: {topic}")
        examples = []
        for i, q in enumerate(questions):
            topic_answer = generate_answer(client, q, topic=topic)
            examples.append(
                {
                    "question": q,
                    "answer": topic_answer,
                    "topic": topic,
                }
            )
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(questions)}")

        slug = topic.lower().replace(" ", "_")
        output_path = data_dir / f"{slug}.json"
        with open(output_path, "w") as f:
            json.dump(examples, f, indent=2)
        print(f"Saved {len(examples)} examples to {output_path}")


if __name__ == "__main__":
    main()

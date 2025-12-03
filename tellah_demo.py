#!.venv/bin/python
"""
Tellah Agent Demo

Shows how to talk to the Tellah-BERT agent programmatically.
"""

import torch
from tellah_bert_trainer import TellahBot

def demo_tellah_conversation():
    """Demonstrate talking to Tellah the Sage."""

    print("🧙 TELLAH-BERT DEMO")
    print("=" * 50)

    # Load the trained Tellah-Bot
    try:
        bot = TellahBot('tellah_bert_model.pth')
        print("✓ Tellah-BERT loaded successfully!")
    except FileNotFoundError:
        print("⚠️  tellah_bert_model.pth not found. Using untrained model.")
        bot = TellahBot()

    print()

    # Sample questions to ask Tellah
    questions = [
        "What is a clock?",
        "How does curvature work?",
        "What are mirror shells?",
        "How do fields emerge from curvature flows?",
        "What is the Galois covering space structure?"
    ]

    for question in questions:
        print(f"You: {question}")

        # Get Tellah's guidance
        result = bot.guide(question)

        print(f"🧙 Tellah (κ={result['curvature']:.2f}, {result['strata']}):")
        print(f"   {result['response']}")
        print(f"   AntClock: x={result['antclock_context']['x']}, R={result['antclock_context']['R']:.3f}")
        print()

    print("✓ Demo complete! Memory contains", result['memory_size'], "exchanges")

def ask_tellah(question: str) -> dict:
    """Helper function to ask Tellah a single question."""
    bot = TellahBot('tellah_bert_model.pth')
    return bot.guide(question)

if __name__ == "__main__":
    demo_tellah_conversation()

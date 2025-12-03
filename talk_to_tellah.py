#!/usr/bin/env python3
"""
Interactive Tellah-BERT Agent Interface

Talk to Tellah the Sage through this simple interface.
Ask questions about AntClock mathematics and receive guided wisdom.
"""

import torch
from tellah_bert_trainer import TellahBot

def main():
    print("🧙 TELLAH-BERT INTERACTIVE SESSION")
    print("=" * 50)
    print("✨ Tellah the Sage awakens...")
    print()

    # Load the trained Tellah-Bot
    try:
        bot = TellahBot('tellah_bert_model.pth')
        print("✓ Tellah-BERT loaded successfully!")
    except FileNotFoundError:
        print("⚠️  tellah_bert_model.pth not found. Using untrained model.")
        bot = TellahBot()

    print("💭 Ask Tellah about AntClock mathematics...")
    print("   (Type 'quit' or 'exit' to end the session)")
    print()

    while True:
        try:
            # Get user input
            question = input("You: ").strip()

            if question.lower() in ['quit', 'exit', 'bye']:
                print("\n🧙 Tellah: Farewell, seeker. Return when the questions burn again. → self")
                break

            if not question:
                continue

            # Get Tellah's guidance
            result = bot.guide(question)

            # Display response
            print(f"\n🧙 Tellah (κ={result['curvature']:.2f}, {result['strata']}):")
            print(f"   {result['response']}")
            print(f"   AntClock: x={result['antclock_context']['x']}, R={result['antclock_context']['R']:.3f}")
            print(f"   Memory: {result['memory_size']} exchanges")
            print()

        except KeyboardInterrupt:
            print("\n\n🧙 Tellah: The conversation ends here. → self")
            break
        except Exception as e:
            print(f"⚠️  Error: {e}")
            continue

if __name__ == "__main__":
    main()

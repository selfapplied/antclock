"""
Enhanced SCAN Dataset Generator

Creates a much larger SCAN-like dataset for proper benchmarking.
"""

import random
from typing import List, Dict, Tuple
from benchmarks.scan import SCANCommand


class EnhancedSCANGenerator:
    """Generates larger SCAN-like datasets for proper benchmarking."""

    def __init__(self):
        # Base actions
        self.actions = ['jump', 'turn_left', 'turn_right', 'walk', 'look', 'run', 'swim', 'crouch']

        # Modifiers
        self.quantifiers = ['', 'twice', 'thrice']
        self.directions = ['', 'left', 'right', 'around']
        self.opposites = ['', 'opposite']

    def generate_atomic_commands(self) -> List[SCANCommand]:
        """Generate basic atomic commands."""
        commands = []

        for action in self.actions:
            # Basic action
            commands.append(SCANCommand(action, action.upper()))

            # With directions (for turn actions)
            if 'turn' in action:
                for direction in ['left', 'right']:
                    cmd = f"{action} {direction}"
                    actions_out = f"{action.upper()}_{direction.upper()}"
                    commands.append(SCANCommand(cmd, actions_out))

        return commands

    def generate_quantified_commands(self) -> List[SCANCommand]:
        """Generate commands with quantifiers (twice, thrice)."""
        commands = []

        for action in self.actions:
            for quantifier in ['twice', 'thrice']:
                if quantifier == 'twice':
                    repeat = 2
                else:  # thrice
                    repeat = 3

                # Simple quantified
                cmd = f"{action} {quantifier}"
                actions_out = ' '.join([action.upper()] * repeat)
                commands.append(SCANCommand(cmd, actions_out))

                # Quantified with direction
                if 'turn' in action:
                    for direction in ['left', 'right']:
                        cmd = f"{action} {direction} {quantifier}"
                        action_base = f"{action}_{direction}".upper()
                        actions_out = ' '.join([action_base] * repeat)
                        commands.append(SCANCommand(cmd, actions_out))

        return commands

    def generate_compound_commands(self) -> List[SCANCommand]:
        """Generate compound commands (and, then, etc.)."""
        commands = []

        for action1 in self.actions[:4]:  # Limit to avoid explosion
            for action2 in self.actions[:4]:
                if action1 != action2:
                    # Simple conjunction
                    cmd = f"{action1} and {action2}"
                    actions_out = f"{action1.upper()} {action2.upper()}"
                    commands.append(SCANCommand(cmd, actions_out))

                    # With quantifier on second action
                    for quantifier in ['twice', 'thrice']:
                        if quantifier == 'twice':
                            repeat = 2
                        else:
                            repeat = 3

                        cmd = f"{action1} and {action2} {quantifier}"
                        actions_out = f"{action1.upper()} {' '.join([action2.upper()] * repeat)}"
                        commands.append(SCANCommand(cmd, actions_out))

        return commands

    def generate_opposite_commands(self) -> List[SCANCommand]:
        """Generate commands with opposites."""
        commands = []

        for action in ['turn_left', 'turn_right']:
            for quantifier in ['', 'twice', 'thrice']:
                if quantifier == 'twice':
                    repeat = 2
                elif quantifier == 'thrice':
                    repeat = 3
                else:
                    repeat = 1

                # Opposite versions
                if action == 'turn_left':
                    opposite = 'turn_right'
                else:
                    opposite = 'turn_left'

                cmd = f"turn opposite {action.split('_')[1]} {quantifier}".strip()
                actions_out = ' '.join([opposite.upper()] * repeat)
                commands.append(SCANCommand(cmd, actions_out))

        return commands

    def generate_systematic_test_set(self) -> List[SCANCommand]:
        """Generate test set with systematic generalization patterns."""
        commands = []

        # Novel combinations not seen in training
        novel_patterns = [
            # Higher-order quantifiers
            ("jump four times", "JUMP JUMP JUMP JUMP"),
            ("turn left four times", "TURN_LEFT TURN_LEFT TURN_LEFT TURN_LEFT"),

            # Complex compounds
            ("run and jump thrice", "RUN JUMP JUMP JUMP"),
            ("walk twice and turn right", "WALK WALK TURN_RIGHT"),

            # Nested opposites
            ("turn opposite left and turn right", "TURN_RIGHT TURN_RIGHT"),
            ("jump and turn opposite right twice", "JUMP TURN_LEFT TURN_LEFT"),

            # Complex directions
            ("turn around left twice", "TURN_LEFT TURN_LEFT TURN_LEFT TURN_LEFT"),
            ("turn opposite around right", "TURN_LEFT TURN_LEFT"),

            # Multi-action sequences
            ("run and walk and jump", "RUN WALK JUMP"),
            ("turn left and jump twice and turn right", "TURN_LEFT JUMP JUMP TURN_RIGHT"),
        ]

        for cmd, actions_out in novel_patterns:
            commands.append(SCANCommand(cmd, actions_out))

        return commands

    def generate_dataset(self, num_train: int = 1000, num_test: int = 200) -> Dict[str, List[SCANCommand]]:
        """Generate complete SCAN-like dataset."""

        # Generate base command sets
        atomic = self.generate_atomic_commands()
        quantified = self.generate_quantified_commands()
        compound = self.generate_compound_commands()
        opposite = self.generate_opposite_commands()

        # Combine for training (subset)
        all_train_commands = atomic + quantified + compound + opposite

        # Extend training set to desired size by variations
        while len(all_train_commands) < num_train:
            # Create variations by combining existing patterns
            if len(all_train_commands) > 10:
                base_cmd = random.choice(all_train_commands[:10])
                # Simple variation: change action
                variation = base_cmd.command.replace('jump', 'run')
                if variation != base_cmd.command:
                    all_train_commands.append(SCANCommand(variation, base_cmd.actions.replace('JUMP', 'RUN')))

        # Limit to desired size
        train_commands = all_train_commands[:num_train]

        # Generate systematic test set
        test_commands = self.generate_systematic_test_set()

        # Extend test set if needed
        while len(test_commands) < num_test and len(all_train_commands) > 20:
            # Add some variations of training patterns
            base = random.choice(all_train_commands[10:20])
            variation = base.command + " again"
            test_commands.append(SCANCommand(variation, base.actions + " " + base.actions.split()[0]))

        test_commands = test_commands[:num_test]

        return {
            'train': train_commands,
            'test': test_commands
        }


def generate_enhanced_scan_dataset():
    """Generate and save enhanced SCAN dataset."""
    generator = EnhancedSCANGenerator()
    dataset = generator.generate_dataset(num_train=500, num_test=100)

    print(f"Generated SCAN dataset:")
    print(f"  Train: {len(dataset['train'])} examples")
    print(f"  Test: {len(dataset['test'])} examples")

    # Save to file
    import json
    with open('enhanced_scan_dataset.json', 'w') as f:
        json.dump({
            'train': [{'command': ex.command, 'actions': ex.actions} for ex in dataset['train']],
            'test': [{'command': ex.command, 'actions': ex.actions} for ex in dataset['test']]
        }, f, indent=2)

    print("Saved to enhanced_scan_dataset.json")
    return dataset


if __name__ == "__main__":
    dataset = generate_enhanced_scan_dataset()
    print("\nSample train examples:")
    for i, ex in enumerate(dataset['train'][:5]):
        print(f"  {ex.command} -> {ex.actions}")

    print("\nSample test examples:")
    for i, ex in enumerate(dataset['test'][:5]):
        print(f"  {ex.command} -> {ex.actions}")

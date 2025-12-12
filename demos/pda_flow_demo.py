#!run.sh
"""
PDA-FLOW Opcode Set Demo

Demonstrates the CE1::PDA-FLOW opcode system with bracket-topology,
morphism generators, and clean operational semantics.
"""

from antclock.pda_flow import (
    PDAFlowState, PDA_FLOW, OpcodeChain,
    OP_FRONTLOAD, OP_SANDWICH, OP_INTERVAL, OP_FLEX,
    OP_BLOCKBACK_CHECK, OP_SLOW, OP_DROP_WEIGHT, OP_RECOVERY,
    resonant_blend, verify_invariants
)


def demo_basic_chain():
    """Demonstrate basic PDA-FLOW chain execution."""
    print("=" * 80)
    print("PDA-FLOW: Basic Chain Execution")
    print("=" * 80)
    
    # Initialize state
    initial_state = PDAFlowState(
        task_queue=["easy_task_1", "easy_task_2", "easy_task_3"],
        hard_task="HARD_TASK",
        task_scope=["subtask1", "subtask2", "subtask3", "subtask4"],
        autonomy_resistance=0.3
    )
    
    print(f"\nInitial State:")
    print(f"  Task queue: {initial_state.task_queue}")
    print(f"  Hard task: {initial_state.hard_task}")
    print(f"  Task scope: {initial_state.task_scope}")
    print(f"  Autonomy resistance: {initial_state.autonomy_resistance:.2f}")
    
    # Create and execute PDA-FLOW chain
    reward = "coffee_break"
    timer = (10.0, 25.0)  # 10-25 minute window
    slack = (5.0, 10.0)   # 5-10 minute slack
    
    chain = PDA_FLOW(reward, timer, slack)
    final_state, witnesses = chain(initial_state)
    
    print(f"\nFinal State:")
    print(f"  Task queue: {final_state.task_queue}")
    print(f"  Reward envelope: before={final_state.reward_before}, after={final_state.reward_after}")
    print(f"  Interval window: {final_state.interval_window}")
    print(f"  Slack window: {final_state.slack_window}")
    print(f"  Autonomy baseline: {final_state.autonomy_baseline:.2f}")
    print(f"  Autonomy resistance: {final_state.autonomy_resistance:.2f}")
    print(f"  Blockback detected: {final_state.blockback_detected}")
    print(f"  Task velocity: {final_state.task_velocity:.2f}")
    print(f"  Task scope size: {len(final_state.task_scope)}")
    print(f"  Residual tension: {final_state.residual_tension:.2f}")
    
    print(f"\nWitnesses ({len(witnesses)} opcodes executed):")
    for i, witness in enumerate(witnesses, 1):
        print(f"  {i}. {witness.opcode_name}")
        print(f"     Effects: {', '.join(witness.effects)}")
        if witness.resonant_frequency is not None:
            print(f"     Resonant frequency: {witness.resonant_frequency:.2f}")
    
    # Verify invariants
    print(f"\nInvariant Verification:")
    invariants = verify_invariants(final_state, witnesses)
    for inv_name, passed in invariants.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {inv_name}: {passed}")
    
    return final_state, witnesses


def demo_blockback_scenario():
    """Demonstrate behavior when blockback is detected."""
    print("\n" + "=" * 80)
    print("PDA-FLOW: Blockback Detection Scenario")
    print("=" * 80)
    
    # Initialize state with high autonomy resistance (will trigger blockback)
    initial_state = PDAFlowState(
        task_queue=["task1", "task2"],
        hard_task="VERY_HARD_TASK",
        task_scope=["subtask1", "subtask2", "subtask3", "subtask4", "subtask5"],
        autonomy_resistance=0.8,  # High resistance
        blockback_threshold=0.5
    )
    
    print(f"\nInitial State (High Resistance):")
    print(f"  Autonomy resistance: {initial_state.autonomy_resistance:.2f}")
    print(f"  Blockback threshold: {initial_state.blockback_threshold:.2f}")
    print(f"  Task scope size: {len(initial_state.task_scope)}")
    
    # Execute chain
    reward = "reward_activity"
    timer = (15.0, 30.0)
    slack = (10.0, 15.0)
    
    chain = PDA_FLOW(reward, timer, slack)
    final_state, witnesses = chain(initial_state)
    
    print(f"\nAfter Execution:")
    print(f"  Blockback detected: {final_state.blockback_detected}")
    print(f"  Final autonomy resistance: {final_state.autonomy_resistance:.2f}")
    print(f"  Task scope size: {len(final_state.task_scope)} (dropped: {len(initial_state.task_scope) - len(final_state.task_scope)})")
    print(f"  Task velocity: {final_state.task_velocity:.2f}")
    
    # Show which opcode was chosen in resonant blend
    for witness in witnesses:
        if "RESONANT_BLEND" in witness.opcode_name:
            print(f"\n  Resonant Blend Choice:")
            print(f"    Frequency: {witness.resonant_frequency:.2f}")
            print(f"    Selected: {witness.opcode_name}")
    
    return final_state, witnesses


def demo_manual_composition():
    """Demonstrate manual opcode composition."""
    print("\n" + "=" * 80)
    print("PDA-FLOW: Manual Opcode Composition")
    print("=" * 80)
    
    initial_state = PDAFlowState(
        task_queue=["task1", "task2", "task3"],
        hard_task="HARD_TASK",
        task_scope=["subtask1", "subtask2"]
    )
    
    # Manually compose opcodes
    chain1 = OpcodeChain([OP_FRONTLOAD(), OP_SANDWICH("reward1")])
    chain2 = OpcodeChain([OP_INTERVAL((10.0, 20.0)), OP_FLEX((5.0, 10.0))])
    
    # Sequential composition
    full_chain = chain1 >> chain2 >> OpcodeChain([OP_RECOVERY()])
    
    final_state, witnesses = full_chain(initial_state)
    
    print(f"\nManual Composition Result:")
    print(f"  Executed {len(witnesses)} opcodes")
    print(f"  Final task queue: {final_state.task_queue}")
    print(f"  Reward envelope: {final_state.reward_before} / {final_state.reward_after}")
    print(f"  Interval: {final_state.interval_window}")
    print(f"  Slack: {final_state.slack_window}")
    
    return final_state, witnesses


def demo_invariant_preservation():
    """Demonstrate that invariants are preserved across execution."""
    print("\n" + "=" * 80)
    print("PDA-FLOW: Invariant Preservation")
    print("=" * 80)
    
    # Test multiple scenarios
    scenarios = [
        ("Low Resistance", PDAFlowState(
            task_queue=["task1"],
            hard_task="HARD_TASK",
            autonomy_resistance=0.2,
            task_scope=["subtask1", "subtask2"]
        )),
        ("Medium Resistance", PDAFlowState(
            task_queue=["task1"],
            hard_task="HARD_TASK",
            autonomy_resistance=0.5,
            task_scope=["subtask1", "subtask2", "subtask3"]
        )),
        ("High Resistance", PDAFlowState(
            task_queue=["task1"],
            hard_task="HARD_TASK",
            autonomy_resistance=0.9,
            task_scope=["subtask1", "subtask2", "subtask3", "subtask4", "subtask5"]
        ))
    ]
    
    for scenario_name, initial_state in scenarios:
        print(f"\n{scenario_name} Scenario:")
        print(f"  Initial resistance: {initial_state.autonomy_resistance:.2f}")
        print(f"  Initial scope size: {len(initial_state.task_scope)}")
        
        chain = PDA_FLOW("reward", (10.0, 25.0), (5.0, 10.0))
        final_state, witnesses = chain(initial_state)
        
        invariants = verify_invariants(final_state, witnesses)
        all_passed = all(invariants.values())
        
        print(f"  Final resistance: {final_state.autonomy_resistance:.2f}")
        print(f"  Final scope size: {len(final_state.task_scope)}")
        print(f"  All invariants: {'✓' if all_passed else '✗'}")
        
        for inv_name, passed in invariants.items():
            if not passed:
                print(f"    ✗ {inv_name} FAILED")


if __name__ == "__main__":
    # Run all demos
    demo_basic_chain()
    demo_blockback_scenario()
    demo_manual_composition()
    demo_invariant_preservation()
    
    print("\n" + "=" * 80)
    print("PDA-FLOW Demo Complete")
    print("=" * 80)
    print("\nKey Features Demonstrated:")
    print("  ✓ Opcode chain composition (>>)")
    print("  ✓ Resonant blend selection (⊕)")
    print("  ✓ Blockback detection and response")
    print("  ✓ Invariant preservation (INV1-INV5)")
    print("  ✓ Reward envelope maintenance")
    print("  ✓ Autonomy-preserving scope reduction")
    print("\nThe system respects autonomy while maintaining throughput! ✨")


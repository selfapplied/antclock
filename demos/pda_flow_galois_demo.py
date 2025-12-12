#!run.sh
"""
PDA-FLOW Galois-Lifted Version Demo

Demonstrates PDA-flow as a field extension F(α) where α = autonomy-index.
Shows how opcodes act as Galois automorphisms preserving field structure.
"""

from antclock.pda_flow_galois import (
    GaloisPDAFlowState, Galois_PDA_FLOW,
    AutonomyIndex, PDAFlowFieldElement,
    GaloisGroup, analyze_field_extension, verify_galois_invariants,
    GaloisOP_FRONTLOAD, GaloisOP_SANDWICH, GaloisOP_INTERVAL,
    GaloisOP_FLEX, GaloisOP_RECOVERY
)
from antclock.pda_flow import PDAFlowState


def demo_field_extension_structure():
    """Demonstrate the field extension F(α) structure."""
    print("=" * 80)
    print("PDA-FLOW Galois Extension: Field Structure")
    print("=" * 80)
    
    # Create initial state
    base_state = PDAFlowState(
        task_queue=["task1", "task2"],
        hard_task="HARD_TASK",
        task_scope=["subtask1", "subtask2", "subtask3"],
        autonomy_baseline=0.8
    )
    
    galois_state = GaloisPDAFlowState(base_state)
    
    print(f"\nBase State:")
    print(f"  Autonomy baseline: {base_state.autonomy_baseline:.2f}")
    print(f"  Task scope size: {len(base_state.task_scope)}")
    print(f"  Task velocity: {base_state.task_velocity:.2f}")
    
    print(f"\nField Extension F(α):")
    element = galois_state.field_element
    print(f"  Field element: {element.base_component:.3f} + {element.autonomy_component:.3f}·α")
    print(f"  Evaluated: {element.evaluate()}")
    
    alpha = galois_state.autonomy_index()
    print(f"\nAutonomy-Index α:")
    print(f"  α = {alpha.value:.3f}")
    print(f"  α̅ = {alpha.conjugate:.3f}")
    print(f"  N(α) = {alpha.norm():.3f}")
    print(f"  Tr(α) = {alpha.trace():.3f}")
    
    print(f"\nField Properties:")
    print(f"  N(state) = {galois_state.field_norm():.3f}")
    print(f"  Tr(state) = {galois_state.field_trace():.3f}")
    
    # Analyze field extension
    analysis = analyze_field_extension(galois_state)
    print(f"\nField Extension Analysis:")
    print(f"  Galois group order: {analysis['galois_group']['order']}")
    print(f"  Isomorphic to: {analysis['galois_group']['isomorphic_to']}")
    print(f"  Orbit size: {analysis['orbit_size']}")
    print(f"  Fixed under identity: {analysis['fixed_fields']['under_identity']}")
    print(f"  Fixed under conjugation: {analysis['fixed_fields']['under_conjugation']}")
    
    return galois_state


def demo_galois_automorphisms():
    """Demonstrate Galois automorphisms acting on states."""
    print("\n" + "=" * 80)
    print("PDA-FLOW Galois Extension: Automorphisms")
    print("=" * 80)
    
    base_state = PDAFlowState(
        task_queue=["task1"],
        hard_task="HARD_TASK",
        autonomy_baseline=0.9
    )
    
    galois_state = GaloisPDAFlowState(base_state)
    galois_group = GaloisGroup()
    
    print(f"\nOriginal State:")
    element = galois_state.field_element
    print(f"  Element: {element.base_component:.3f} + {element.autonomy_component:.3f}·α")
    print(f"  Evaluated: {element.evaluate()}")
    print(f"  Norm: {galois_state.field_norm():.3f}")
    
    # Apply identity automorphism
    print(f"\nIdentity Automorphism (id):")
    id_state = galois_state.apply_automorphism(galois_group.identity)
    print(f"  Result: {id_state.field_element.base_component:.3f} + {id_state.field_element.autonomy_component:.3f}·α")
    print(f"  Norm preserved: {abs(galois_state.field_norm() - id_state.field_norm()) < 1e-10}")
    
    # Apply conjugation automorphism
    print(f"\nConjugation Automorphism (conj):")
    conj_state = galois_state.apply_automorphism(galois_group.conjugation)
    print(f"  Result: {conj_state.field_element.base_component:.3f} + {conj_state.field_element.autonomy_component:.3f}·α̅")
    print(f"  Norm preserved: {abs(galois_state.field_norm() - conj_state.field_norm()) < 1e-10}")
    print(f"  Autonomy baseline: {conj_state.base_state.autonomy_baseline:.3f}")
    
    # Show orbit
    print(f"\nGalois Orbit:")
    orbit = galois_group.orbit(galois_state.field_element)
    for i, orbit_element in enumerate(orbit):
        print(f"  {i+1}. {orbit_element.base_component:.3f} + {orbit_element.autonomy_component:.3f}·α{'̅' if i > 0 else ''}")
    
    return galois_state, galois_group


def demo_galois_lifted_opcodes():
    """Demonstrate Galois-lifted opcodes as automorphisms."""
    print("\n" + "=" * 80)
    print("PDA-FLOW Galois Extension: Lifted Opcodes")
    print("=" * 80)
    
    base_state = PDAFlowState(
        task_queue=["task1", "task2", "task3"],
        hard_task="HARD_TASK",
        task_scope=["subtask1", "subtask2", "subtask3", "subtask4"],
        autonomy_baseline=0.85,
        autonomy_resistance=0.3
    )
    
    galois_state = GaloisPDAFlowState(base_state)
    
    print(f"\nInitial State:")
    print(f"  Autonomy baseline: {galois_state.base_state.autonomy_baseline:.2f}")
    print(f"  Field norm: {galois_state.field_norm():.3f}")
    print(f"  Autonomy-index: {galois_state.autonomy_index().value:.3f}")
    
    # Execute Galois-lifted opcodes
    opcodes = [
        ("FRONTLOAD", GaloisOP_FRONTLOAD()),
        ("SANDWICH", GaloisOP_SANDWICH("reward")),
        ("INTERVAL", GaloisOP_INTERVAL((10.0, 25.0))),
        ("FLEX", GaloisOP_FLEX((5.0, 10.0))),
        ("RECOVERY", GaloisOP_RECOVERY())
    ]
    
    current_state = galois_state
    print(f"\nExecuting Galois-Lifted Opcodes:")
    
    for opcode_name, opcode in opcodes:
        current_state, witness = opcode(current_state)
        print(f"\n  {opcode_name}:")
        print(f"    Autonomy baseline: {current_state.base_state.autonomy_baseline:.3f}")
        print(f"    Field norm: {current_state.field_norm():.3f}")
        print(f"    Autonomy-index: {current_state.autonomy_index().value:.3f}")
        print(f"    Effects: {', '.join(witness.effects[-2:])}")  # Show Galois effects
    
    return current_state


def demo_complete_galois_chain():
    """Demonstrate complete Galois-lifted PDA-FLOW chain."""
    print("\n" + "=" * 80)
    print("PDA-FLOW Galois Extension: Complete Chain")
    print("=" * 80)
    
    base_state = PDAFlowState(
        task_queue=["task1", "task2"],
        hard_task="VERY_HARD_TASK",
        task_scope=["subtask1", "subtask2", "subtask3", "subtask4", "subtask5"],
        autonomy_baseline=0.9,
        autonomy_resistance=0.6  # Moderate resistance
    )
    
    galois_state = GaloisPDAFlowState(base_state)
    
    print(f"\nInitial State:")
    print(f"  Autonomy baseline: {galois_state.base_state.autonomy_baseline:.2f}")
    print(f"  Autonomy resistance: {galois_state.base_state.autonomy_resistance:.2f}")
    print(f"  Field norm: {galois_state.field_norm():.3f}")
    
    # Execute complete Galois-lifted chain
    chain = Galois_PDA_FLOW("reward", (15.0, 30.0), (10.0, 15.0))
    final_state, witnesses = chain(galois_state)
    
    print(f"\nFinal State:")
    print(f"  Autonomy baseline: {final_state.base_state.autonomy_baseline:.2f}")
    print(f"  Autonomy resistance: {final_state.base_state.autonomy_resistance:.2f}")
    print(f"  Field norm: {final_state.field_norm():.3f}")
    print(f"  Autonomy-index: {final_state.autonomy_index().value:.3f}")
    print(f"  Task scope size: {len(final_state.base_state.task_scope)}")
    
    print(f"\nWitnesses ({len(witnesses)} opcodes):")
    for i, witness in enumerate(witnesses, 1):
        galois_effects = [e for e in witness.effects if "Galois:" in e]
        if galois_effects:
            print(f"  {i}. {witness.opcode_name}: {', '.join(galois_effects)}")
    
    # Verify Galois invariants
    print(f"\nGalois Invariant Verification:")
    invariants = verify_galois_invariants(final_state)
    for inv_name, passed in invariants.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {inv_name}: {passed}")
    
    return final_state, witnesses


def demo_field_extension_comparison():
    """Compare states in base field vs. extended field."""
    print("\n" + "=" * 80)
    print("PDA-FLOW Galois Extension: Base vs. Extended Field")
    print("=" * 80)
    
    # Create states with different autonomy baselines
    states = [
        ("Low Autonomy", PDAFlowState(
            task_queue=["task1"],
            hard_task="HARD_TASK",
            autonomy_baseline=0.3
        )),
        ("Medium Autonomy", PDAFlowState(
            task_queue=["task1"],
            hard_task="HARD_TASK",
            autonomy_baseline=0.7
        )),
        ("High Autonomy", PDAFlowState(
            task_queue=["task1"],
            hard_task="HARD_TASK",
            autonomy_baseline=1.0
        ))
    ]
    
    print(f"\nField Extension Comparison:")
    
    for state_name, base_state in states:
        galois_state = GaloisPDAFlowState(base_state)
        element = galois_state.field_element
        alpha = galois_state.autonomy_index()
        
        print(f"\n{state_name}:")
        print(f"  Base field: throughput = {element.base_component:.3f}")
        print(f"  Extended field: {element.base_component:.3f} + {element.autonomy_component:.3f}·α")
        print(f"  α = {alpha.value:.3f}")
        print(f"  N(α) = {alpha.norm():.3f}")
        print(f"  N(state) = {galois_state.field_norm():.3f}")
        
        # Show how autonomy-index extends the field
        if alpha.norm() < 0:
            print(f"  Extension type: Real (α² = {alpha.value**2:.3f} < 0 in base field)")
        else:
            print(f"  Extension type: Complex (α² = {alpha.value**2:.3f} ≥ 0)")


if __name__ == "__main__":
    # Run all demos
    demo_field_extension_structure()
    demo_galois_automorphisms()
    demo_galois_lifted_opcodes()
    demo_complete_galois_chain()
    demo_field_extension_comparison()
    
    print("\n" + "=" * 80)
    print("PDA-FLOW Galois Extension Demo Complete")
    print("=" * 80)
    print("\nKey Features Demonstrated:")
    print("  ✓ Field extension F(α) where α = autonomy-index")
    print("  ✓ Galois group Gal(F(α)/F) ≅ C₂")
    print("  ✓ Opcodes as field automorphisms")
    print("  ✓ Field norm and trace preservation")
    print("  ✓ Galois invariants (INV_GALOIS_1, INV_GALOIS_2, INV_GALOIS_3)")
    print("  ✓ Autonomy as first-class field element")
    print("\nAutonomy is now a field element, not just a constraint! ✨")


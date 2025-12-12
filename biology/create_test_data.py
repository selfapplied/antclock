#!/usr/bin/env python3
"""Create synthetic ERV test data for pipeline testing."""
import random
from pathlib import Path

def generate_erv_sequence(length=500, gc_content=0.45):
    """Generate synthetic ERV-like sequence."""
    bases = ['A', 'T', 'G', 'C']
    gc_bases = ['G', 'C']
    at_bases = ['A', 'T']
    
    seq = []
    for i in range(length):
        if random.random() < gc_content:
            seq.append(random.choice(gc_bases))
        else:
            seq.append(random.choice(at_bases))
    return ''.join(seq)

def create_test_fasta(output_file: Path, num_sequences=5):
    """Create test FASTA file with synthetic ERV sequences."""
    with open(output_file, 'w') as f:
        for i in range(num_sequences):
            seq_id = f"test_erv_{i+1}"
            length = random.randint(300, 800)
            sequence = generate_erv_sequence(length)
            f.write(f">{seq_id} synthetic ERV sequence for testing\n")
            f.write(sequence + "\n")
    print(f"✅ Created {num_sequences} test sequences in {output_file}")

if __name__ == '__main__':
    data_dir = Path(__file__).parent / "data" / "test"
    data_dir.mkdir(parents=True, exist_ok=True)
    output = data_dir / "test_erv_sequences.fasta"
    create_test_fasta(output, num_sequences=5)






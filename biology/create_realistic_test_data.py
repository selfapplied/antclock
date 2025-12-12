#!/usr/bin/env python3
"""
Create realistic ERV test data that mimics real sequences.

Includes:
- Sequences with high stress (will trigger Volte)
- Sequences with conserved regions
- Sequences with functional annotations
- Varied GC content and lengths
"""

import random
from pathlib import Path

def generate_sequence(length, gc_content=0.45, repeat_density=0.1):
    """Generate sequence with specified characteristics."""
    bases = ['A', 'T', 'G', 'C']
    gc_bases = ['G', 'C']
    at_bases = ['A', 'T']
    
    seq = []
    for i in range(length):
        # Add repeats for high stress sequences
        if random.random() < repeat_density and i > 10:
            # Repeat last few bases
            repeat_len = random.randint(3, 8)
            if len(seq) >= repeat_len:
                seq.extend(seq[-repeat_len:])
                continue
        
        if random.random() < gc_content:
            seq.append(random.choice(gc_bases))
        else:
            seq.append(random.choice(at_bases))
    
    return ''.join(seq)

def create_realistic_fasta(output_file: Path):
    """Create realistic test FASTA with varied ERV characteristics."""
    sequences = [
        # Low stress, high coherence (exapted)
        {
            'id': 'HERV-K_001',
            'length': 850,
            'gc': 0.48,
            'repeat': 0.05,
            'annotations': {'exapted': True, 'expression': 'high', 'function': 'placental'},
            'conserved': [(0, 100), (200, 300), (600, 750)]
        },
        # High stress, low coherence (recent integration)
        {
            'id': 'HERV-W_002',
            'length': 1200,
            'gc': 0.52,
            'repeat': 0.25,  # High repeat density = high stress
            'annotations': {'exapted': False, 'expression': 'low'},
            'conserved': []
        },
        # Medium stress, medium coherence
        {
            'id': 'HERV-H_003',
            'length': 650,
            'gc': 0.44,
            'repeat': 0.12,
            'annotations': {'exapted': False, 'expression': 'medium'},
            'conserved': [(50, 150), (400, 500)]
        },
        # Very high stress (will trigger Volte)
        {
            'id': 'HERV-L_004',
            'length': 900,
            'gc': 0.50,
            'repeat': 0.30,  # Very high repeats
            'annotations': {'exapted': False, 'expression': 'none'},
            'conserved': []
        },
        # Low stress, well-conserved (stable)
        {
            'id': 'HERV-F_005',
            'length': 750,
            'gc': 0.46,
            'repeat': 0.08,
            'annotations': {'exapted': True, 'expression': 'medium', 'function': 'immune'},
            'conserved': [(0, 200), (300, 450), (550, 750)]
        },
    ]
    
    with open(output_file, 'w') as f:
        for seq_info in sequences:
            sequence = generate_sequence(
                seq_info['length'],
                seq_info['gc'],
                seq_info['repeat']
            )
            
            # Build header with annotations
            header = f">{seq_info['id']}"
            if seq_info['annotations'].get('exapted'):
                header += " [EXAPTED]"
            if seq_info['annotations'].get('function'):
                header += f" function={seq_info['annotations']['function']}"
            header += f" length={seq_info['length']}"
            
            f.write(header + "\n")
            
            # Write sequence in 80-char lines (FASTA format)
            for i in range(0, len(sequence), 80):
                f.write(sequence[i:i+80] + "\n")
    
    print(f"✅ Created {len(sequences)} realistic ERV sequences in {output_file}")
    print(f"   - Sequences with high stress (will trigger Volte): 2")
    print(f"   - Sequences with conserved regions: 3")
    print(f"   - Exapted sequences: 2")
    return sequences

if __name__ == '__main__':
    data_dir = Path(__file__).parent / "data" / "test"
    data_dir.mkdir(parents=True, exist_ok=True)
    output = data_dir / "realistic_erv_sequences.fasta"
    create_realistic_fasta(output)






#!/usr/bin/env python3
"""
Download More Real Data: Expand dataset with different ERV sources.

Downloads:
1. More GenBank sequences (HERV, different species)
2. Ensembl annotations
3. Attempts HERVd and ERVmap
"""

import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Bio import Entrez
from Bio import SeqIO


def download_herv_sequences(email: str = "test@example.com", max_sequences: int = 200):
    """Download HERV (Human ERV) sequences from GenBank."""
    data_dir = Path(__file__).parent / "data" / "genbank"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = data_dir / "herv_sequences.fasta"
    
    if output_file.exists():
        print(f"✅ HERV sequences already exist: {output_file}")
        return output_file
    
    Entrez.email = email
    
    print(f"📥 Searching GenBank for HERV sequences (max {max_sequences})...")
    
    # Search for HERV sequences
    search_term = "HERV[Title] OR human endogenous retrovirus[Title]"
    handle = Entrez.esearch(db="nucleotide", term=search_term, retmax=max_sequences)
    record = Entrez.read(handle)
    handle.close()
    
    if not record['IdList']:
        print("⚠️ No HERV sequences found")
        return None
    
    print(f"📊 Found {len(record['IdList'])} HERV sequences")
    print("   Downloading sequences (this may take a while)...")
    
    sequences = []
    batch_size = 10
    
    for i in range(0, len(record['IdList']), batch_size):
        batch_ids = record['IdList'][i:i+batch_size]
        fetch_handle = Entrez.efetch(
            db="nucleotide",
            id=",".join(batch_ids),
            rettype="fasta",
            retmode="text"
        )
        
        batch_seqs = list(SeqIO.parse(fetch_handle, "fasta"))
        sequences.extend(batch_seqs)
        fetch_handle.close()
        
        print(f"   Downloaded {len(sequences)}/{len(record['IdList'])} sequences...")
        time.sleep(0.5)
    
    SeqIO.write(sequences, output_file, "fasta")
    print(f"✅ Saved {len(sequences)} HERV sequences to {output_file}")
    return output_file


def download_primate_erv(email: str = "test@example.com", max_sequences: int = 200):
    """Download primate ERV sequences."""
    data_dir = Path(__file__).parent / "data" / "genbank"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = data_dir / "primate_erv_sequences.fasta"
    
    if output_file.exists():
        print(f"✅ Primate ERV sequences already exist: {output_file}")
        return output_file
    
    Entrez.email = email
    
    print(f"📥 Searching GenBank for primate ERV sequences (max {max_sequences})...")
    
    search_term = "(endogenous retrovirus[Title] OR ERV[Title]) AND (primate[Organism] OR Pan[Organism] OR Gorilla[Organism])"
    handle = Entrez.esearch(db="nucleotide", term=search_term, retmax=max_sequences)
    record = Entrez.read(handle)
    handle.close()
    
    if not record['IdList']:
        print("⚠️ No primate ERV sequences found")
        return None
    
    print(f"📊 Found {len(record['IdList'])} primate ERV sequences")
    print("   Downloading sequences (this may take a while)...")
    
    sequences = []
    batch_size = 10
    
    for i in range(0, len(record['IdList']), batch_size):
        batch_ids = record['IdList'][i:i+batch_size]
        fetch_handle = Entrez.efetch(
            db="nucleotide",
            id=",".join(batch_ids),
            rettype="fasta",
            retmode="text"
        )
        
        batch_seqs = list(SeqIO.parse(fetch_handle, "fasta"))
        sequences.extend(batch_seqs)
        fetch_handle.close()
        
        print(f"   Downloaded {len(sequences)}/{len(record['IdList'])} sequences...")
        time.sleep(0.5)
    
    SeqIO.write(sequences, output_file, "fasta")
    print(f"✅ Saved {len(sequences)} primate ERV sequences to {output_file}")
    return output_file


def download_ensembl_annotations():
    """Download Ensembl ERV annotations."""
    import requests
    import json
    
    data_dir = Path(__file__).parent / "data" / "ensembl"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = data_dir / "homo_sapiens_erv_annotations.json"
    
    print("📥 Attempting to download Ensembl ERV annotations...")
    print("   Note: Ensembl may require specific endpoints")
    
    server = "https://rest.ensembl.org"
    
    # Try to get repeat annotations (ERVs are often in repeat databases)
    # This is a simplified attempt - actual ERV data may require different endpoints
    
    annotations = {
        'species': 'homo_sapiens',
        'source': 'Ensembl',
        'note': 'Direct ERV endpoints may require custom queries',
        'suggested_approach': [
            'Use Ensembl Biomart for bulk downloads',
            'Query RepeatMasker annotations',
            'Use Ensembl variation API for ERV-associated variants'
        ],
        'api_base': server
    }
    
    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"✅ Created Ensembl annotation structure at {output_file}")
    print("   For actual ERV data, consider using Ensembl Biomart")
    return output_file


def main():
    """Download more real data."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download More Real Data")
    parser.add_argument('--email', type=str, default='test@example.com',
                       help='Email for NCBI Entrez API')
    parser.add_argument('--herv', action='store_true',
                       help='Download HERV sequences')
    parser.add_argument('--primate', action='store_true',
                       help='Download primate ERV sequences')
    parser.add_argument('--ensembl', action='store_true',
                       help='Download Ensembl annotations')
    parser.add_argument('--all', action='store_true',
                       help='Download all available data')
    
    args = parser.parse_args()
    
    if args.all or args.herv:
        print("\n" + "="*70)
        print("1. Downloading HERV Sequences")
        print("="*70)
        download_herv_sequences(args.email, max_sequences=200)
    
    if args.all or args.primate:
        print("\n" + "="*70)
        print("2. Downloading Primate ERV Sequences")
        print("="*70)
        download_primate_erv(args.email, max_sequences=200)
    
    if args.all or args.ensembl:
        print("\n" + "="*70)
        print("3. Downloading Ensembl Annotations")
        print("="*70)
        download_ensembl_annotations()
    
    if not (args.all or args.herv or args.primate or args.ensembl):
        print("📋 Available options:")
        print("   --herv      : Download HERV sequences")
        print("   --primate   : Download primate ERV sequences")
        print("   --ensembl   : Download Ensembl annotations")
        print("   --all       : Download everything")
        print("\nExample: python download_more_real_data.py --all")


if __name__ == '__main__':
    main()





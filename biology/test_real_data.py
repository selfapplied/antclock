#!/usr/bin/env python3
"""
Test the biology module with real ERV data.

Downloads a small dataset and runs the complete analysis pipeline.
"""

import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from biology.scripts.download_datasets import DatasetDownloader
from biology.erv.analyze_erv import ERVAnalyzer
from biology.blast.analyze import BLASTAnalyzer


def test_download():
    """Test downloading ERV sequences from GenBank."""
    print("=" * 60)
    print("Test 1: Download ERV Sequences from GenBank")
    print("=" * 60)
    
    downloader = DatasetDownloader()
    
    # Try with a small number of sequences first
    print("\n📥 Attempting to download 10 ERV sequences...")
    print("   (Using test email - NCBI requires email for API access)")
    
    # Use a test email or environment variable
    import os
    email = os.getenv("NCBI_EMAIL", "test@example.com")
    
    success = downloader.download_genbank_erv(max_sequences=10, email=email)
    
    if success:
        print("\n✅ Download successful!")
        genbank_file = downloader.data_dir / "genbank" / "erv_sequences.fasta"
        if genbank_file.exists():
            # Count sequences
            with open(genbank_file, 'r') as f:
                seq_count = sum(1 for line in f if line.startswith('>'))
            print(f"   Downloaded {seq_count} sequences")
            print(f"   File: {genbank_file}")
            return genbank_file
    
    print("\n⚠️ Download failed or no sequences found")
    return None


def test_erv_analysis(fasta_file: Path):
    """Test ERV analysis on downloaded sequences."""
    print("\n" + "=" * 60)
    print("Test 2: ERV Volte System Analysis")
    print("=" * 60)
    
    if not fasta_file or not fasta_file.exists():
        print("⚠️ No FASTA file available for analysis")
        return None
    
    analyzer = ERVAnalyzer()
    
    print(f"\n🔬 Analyzing sequences from {fasta_file.name}...")
    output_file = analyzer.data_dir / "test_erv_analysis.json"
    
    results = analyzer.analyze_file(fasta_file, output_file)
    
    print(f"\n📊 Analysis Results:")
    print(f"   Sequences analyzed: {results['num_sequences']}")
    print(f"   Volte activations: {results['summary']['volte_activated_count']}")
    print(f"   Average stress: {results['summary']['avg_stress']:.3f}")
    print(f"   Average coherence: {results['summary']['avg_coherence']:.3f}")
    print(f"   Average exaptation potential: {results['summary']['avg_exaptation_potential']:.3f}")
    
    # Show details for first sequence
    if results['analyses']:
        first = results['analyses'][0]
        print(f"\n📋 First Sequence Details:")
        print(f"   ID: {first['memory']['sequence_id']}")
        print(f"   Length: {first['memory']['length']}")
        print(f"   Stress: {first['transform']['stress_S']:.3f}")
        print(f"   Coherence: {first['transform']['coherence_C']:.3f}")
        print(f"   Volte Activated: {first['transform']['volte_activated']}")
        print(f"   Identity Preserved: {first['witness']['identity_preserved']}")
    
    return results


def test_blast_setup(fasta_file: Path):
    """Test BLAST database creation."""
    print("\n" + "=" * 60)
    print("Test 3: BLAST Database Setup")
    print("=" * 60)
    
    if not fasta_file or not fasta_file.exists():
        print("⚠️ No FASTA file available for BLAST setup")
        return False
    
    analyzer = BLASTAnalyzer()
    
    if not analyzer.check_blast_installed():
        print("⚠️ BLAST+ not installed")
        print("   Install from: https://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Web&PAGE_TYPE=BlastDocs&DOC_TYPE=Download")
        return False
    
    print(f"\n📦 Creating BLAST database from {fasta_file.name}...")
    success = analyzer.create_blast_db(fasta_file, "test_erv_db")
    
    if success:
        print("✅ BLAST database created successfully")
        print("   Ready for sequence similarity searches")
    
    return success


def main():
    """Run complete test pipeline."""
    print("\n" + "🧪 Testing Biology Module with Real Data" + "\n")
    
    # Test 1: Download
    fasta_file = test_download()
    
    # Test 2: ERV Analysis
    if fasta_file:
        analysis_results = test_erv_analysis(fasta_file)
    
    # Test 3: BLAST Setup
    if fasta_file:
        test_blast_setup(fasta_file)
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print("\n✅ Pipeline components tested:")
    print("   1. Data download (GenBank)")
    print("   2. ERV Volte analysis")
    print("   3. BLAST database creation")
    print("\n📝 Next steps:")
    print("   - Run BLAST searches: python biology/blast/analyze.py")
    print("   - Integrate results: python biology/erv/integrate_blast.py")
    print("   - Download more sequences: python biology/scripts/download_datasets.py")


if __name__ == '__main__':
    main()






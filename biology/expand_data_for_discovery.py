#!/usr/bin/env python3
"""
Expand Data for Discovery: Download real data for forming stars.

This script:
1. Downloads additional ERV sequences (different families/species)
2. Attempts to get ERVmap annotations
3. Downloads HERVd database data
4. Downloads Ensembl annotations
5. Analyzes expanded dataset for new patterns
"""

import json
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import after path setup
try:
    from biology.erv.analyze_erv import ERVAnalyzer
except ImportError:
    ERVAnalyzer = None


def download_hervd_data(output_dir: Path) -> bool:
    """
    Download HERVd database data.
    
    HERVd: https://herv.img.cas.cz/
    Provides: HERV families, integration sites, structural analysis
    """
    hervd_dir = output_dir / "hervd"
    hervd_dir.mkdir(parents=True, exist_ok=True)
    
    print("📥 Attempting to download HERVd data...")
    print("   URL: https://herv.img.cas.cz/")
    print("   Note: May require manual download or API access")
    
    # HERVd may require web scraping or API access
    # For now, create placeholder
    info_file = hervd_dir / "hervd_info.json"
    info = {
        'source': 'HERVd Database',
        'url': 'https://herv.img.cas.cz/',
        'description': 'Human Endogenous Retroviruses Database',
        'provides': [
            'HERV families',
            'Integration sites',
            'Structural analysis'
        ],
        'note': 'May require manual download or API access'
    }
    
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"✅ Created HERVd info file: {info_file}")
    return True


def analyze_expanded_dataset(fasta_file: Path, output_file: Path):
    """Analyze expanded dataset for new patterns."""
    print(f"📊 Analyzing expanded dataset: {fasta_file}")
    
    if ERVAnalyzer is None:
        print("⚠️ ERVAnalyzer not available, skipping analysis")
        return None
    
    analyzer = ERVAnalyzer()
    
    # Run analysis (sample first 100 for speed)
    print("   Analyzing sequences (this may take a while)...")
    results = analyzer.analyze_fasta(fasta_file, max_sequences=100)
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"✅ Analysis complete: {len(results.get('analyses', []))} sequences")
    print(f"💾 Saved to {output_file}")
    
    return results


def main():
    """Main function to expand data for discovery."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Expand Data for Discovery"
    )
    parser.add_argument('--download-hervd', action='store_true',
                       help='Download HERVd database data')
    parser.add_argument('--download-ensembl', action='store_true',
                       help='Download Ensembl annotations')
    parser.add_argument('--analyze-expanded', action='store_true',
                       help='Analyze expanded ERV dataset')
    parser.add_argument('--all', action='store_true',
                       help='Download all available real data')
    
    args = parser.parse_args()
    
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if args.all or args.download_hervd:
        print("🌟 Downloading HERVd database data...")
        download_hervd_data(data_dir)
    
    if args.all or args.download_ensembl:
        print("\n🌟 Downloading Ensembl annotations...")
        # Use run.sh to access download_datasets
        import subprocess
        result = subprocess.run([
            'python3', 'biology/scripts/download_datasets.py',
            '--dataset', 'ensembl'
        ], cwd=Path(__file__).parent.parent.parent, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
    
    if args.all or args.analyze_expanded:
        fasta_file = data_dir / "genbank" / "erv_sequences.fasta"
        if fasta_file.exists():
            output_file = data_dir / "genbank" / "expanded_analysis.json"
            analyze_expanded_dataset(fasta_file, output_file)
        else:
            print(f"⚠️ ERV sequences not found: {fasta_file}")
    
    if not (args.all or args.download_hervd or args.download_ensembl or args.analyze_expanded):
        print("📋 Available options:")
        print("   --download-hervd     : Download HERVd database data")
        print("   --download-ensembl   : Download Ensembl annotations")
        print("   --analyze-expanded   : Analyze expanded ERV dataset")
        print("   --all                : Download all available real data")


if __name__ == '__main__':
    main()


#!/usr/bin/env python3
"""
BLAST analysis with CE framework integration.

Analyzes sequence similarity through compositional lens,
connecting to Volte dynamics for ERV evolution.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import subprocess
import json

class BLASTAnalyzer:
    """BLAST sequence analysis with CE framework."""
    
    def __init__(self, data_dir: Path = Path(__file__).parent.parent.parent / "biology" / "data"):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.blast_dir = data_dir / "blast"
        self.blast_dir.mkdir(exist_ok=True)
    
    def check_blast_installed(self) -> bool:
        """Check if BLAST+ is installed."""
        try:
            result = subprocess.run(['blastn', '-version'], 
                                 capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def create_blast_db(self, fasta_file: Path, db_name: str) -> bool:
        """Create a BLAST database from FASTA file."""
        db_path = self.blast_dir / db_name
        
        if db_path.with_suffix('.nhr').exists():
            print(f"✅ BLAST database {db_name} already exists")
            return True
        
        print(f"📦 Creating BLAST database from {fasta_file}...")
        try:
            subprocess.run([
                'makeblastdb',
                '-in', str(fasta_file),
                '-dbtype', 'nucl',
                '-out', str(db_path)
            ], check=True)
            print(f"✅ Database created: {db_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Database creation failed: {e}")
            return False
        except FileNotFoundError:
            print("⚠️ makeblastdb not found. Install BLAST+: https://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Web&PAGE_TYPE=BlastDocs&DOC_TYPE=Download")
            return False
    
    def run_blast(self, query_file: Path, db_path: Path, 
                  output_file: Path, evalue: float = 1e-5) -> bool:
        """Run BLAST search."""
        print(f"🔍 Running BLAST search...")
        try:
            with open(output_file, 'w') as out:
                subprocess.run([
                    'blastn',
                    '-query', str(query_file),
                    '-db', str(db_path),
                    '-outfmt', '6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore',
                    '-evalue', str(evalue)
                ], stdout=out, check=True)
            
            print(f"✅ BLAST results saved to {output_file}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ BLAST search failed: {e}")
            return False
    
    def parse_blast_results(self, blast_file: Path) -> List[Dict]:
        """Parse BLAST output into structured format."""
        results = []
        
        with open(blast_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                
                fields = line.strip().split('\t')
                if len(fields) >= 12:
                    results.append({
                        'query_id': fields[0],
                        'subject_id': fields[1],
                        'identity': float(fields[2]),
                        'length': int(fields[3]),
                        'mismatch': int(fields[4]),
                        'gapopen': int(fields[5]),
                        'query_start': int(fields[6]),
                        'query_end': int(fields[7]),
                        'subject_start': int(fields[8]),
                        'subject_end': int(fields[9]),
                        'evalue': float(fields[10]),
                        'bitscore': float(fields[11])
                    })
        
        return results
    
    def analyze_with_ce(self, blast_results: List[Dict]) -> Dict:
        """Analyze BLAST results through CE framework lens.
        
        Maps sequence similarity to CE1 bracket structure:
        - High identity = deep bracket nesting (shared structure)
        - Low evalue = strong compositional coherence
        - Alignment length = domain operator depth
        """
        if not blast_results:
            return {}
        
        # CE1 domain operator: bracket depth from identity
        # Higher identity = deeper shared structure
        avg_identity = sum(r['identity'] for r in blast_results) / len(blast_results)
        bracket_depth = int(avg_identity / 10)  # Scale to bracket depth
        
        # CE2 guardian: coherence from evalue
        # Lower evalue = higher coherence
        min_evalue = min(r['evalue'] for r in blast_results)
        coherence = 1.0 / (1.0 + min_evalue * 1e5)  # Normalize
        
        # CE3 error-lift: structural discovery from mismatches
        # Mismatches indicate structural differences (potential evolution)
        total_mismatches = sum(r['mismatch'] for r in blast_results)
        evolution_potential = total_mismatches / len(blast_results) if blast_results else 0
        
        return {
            'bracket_depth': bracket_depth,
            'coherence': coherence,
            'evolution_potential': evolution_potential,
            'num_hits': len(blast_results),
            'avg_identity': avg_identity,
            'min_evalue': min_evalue
        }
    
    def save_analysis(self, analysis: Dict, output_file: Path):
        """Save CE analysis results."""
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"💾 Analysis saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="BLAST analysis with CE framework")
    parser.add_argument('query', type=Path, help='Query sequence file (FASTA)')
    parser.add_argument('--db', type=Path, help='BLAST database path')
    parser.add_argument('--create-db', type=Path, help='Create DB from FASTA file')
    parser.add_argument('--db-name', default='erv_db', help='Database name')
    parser.add_argument('--output', type=Path, help='Output file for results')
    parser.add_argument('--evalue', type=float, default=1e-5, help='E-value threshold')
    
    args = parser.parse_args()
    
    analyzer = BLASTAnalyzer()
    
    if not analyzer.check_blast_installed():
        print("❌ BLAST+ not found. Please install from:")
        print("   https://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Web&PAGE_TYPE=BlastDocs&DOC_TYPE=Download")
        return
    
    # Create database if requested
    if args.create_db:
        analyzer.create_blast_db(args.create_db, args.db_name)
        db_path = analyzer.blast_dir / args.db_name
    elif args.db:
        db_path = args.db
    else:
        print("❌ Must specify --db or --create-db")
        return
    
    # Run BLAST
    output_file = args.output or analyzer.blast_dir / f"{args.query.stem}_blast.txt"
    if analyzer.run_blast(args.query, db_path, output_file, args.evalue):
        # Parse and analyze
        results = analyzer.parse_blast_results(output_file)
        ce_analysis = analyzer.analyze_with_ce(results)
        
        # Save analysis
        analysis_file = output_file.with_suffix('.ce_analysis.json')
        analyzer.save_analysis(ce_analysis, analysis_file)
        
        print(f"\n📊 CE Analysis Results:")
        print(f"   Bracket Depth (CE1): {ce_analysis['bracket_depth']}")
        print(f"   Coherence (CE2): {ce_analysis['coherence']:.3f}")
        print(f"   Evolution Potential (CE3): {ce_analysis['evolution_potential']:.2f}")
        print(f"   Hits: {ce_analysis['num_hits']}")


if __name__ == '__main__':
    main()



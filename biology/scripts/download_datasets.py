"""
Download biological datasets for ERV research.

Downloads from:
- HERVd database
- GenBank ERV sequences
- Ensembl annotations
- RepeatMasker data
"""

import argparse
import urllib.request
import zipfile
import gzip
import json
from pathlib import Path
from typing import Dict, List, Optional
import time

class DatasetDownloader:
    """Manages download of biological datasets."""
    
    def __init__(self, data_dir: Path = Path(__file__).parent.parent / "data"):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.datasets = {
            'hervd': {
                'name': 'HERVd Database',
                'url': 'https://herv.img.cas.cz/',
                'type': 'web_scrape',  # May need API access
                'description': 'Human Endogenous Retroviruses Database'
            },
            'genbank_erv': {
                'name': 'GenBank ERV Sequences',
                'url': 'https://www.ncbi.nlm.nih.gov/genbank/',
                'type': 'entrez',  # Requires Entrez API
                'description': 'ERV sequences from GenBank'
            },
            'ensembl_erv': {
                'name': 'Ensembl ERV Annotations',
                'url': 'https://www.ensembl.org/',
                'type': 'api',  # Requires Ensembl API
                'description': 'ERV annotations from Ensembl'
            },
            'repeatmasker': {
                'name': 'RepeatMasker Database',
                'url': 'http://www.repeatmasker.org/',
                'type': 'download',
                'description': 'Repeat element database'
            },
            'ervmap': {
                'name': 'ERVmap',
                'url': 'https://github.com/Functional-Genomics/ERVmap',
                'type': 'git',
                'description': 'Human ERV annotations and expression'
            }
        }
    
    def download_file(self, url: str, dest_path: Path, description: str = "") -> bool:
        """Download a file from URL."""
        if dest_path.exists():
            print(f"✅ {description or dest_path.name} already exists")
            return True
        
        print(f"📥 Downloading {description or url}...")
        try:
            urllib.request.urlretrieve(url, dest_path)
            print(f"✅ Download complete: {dest_path}")
            return True
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False
    
    def download_ervmap(self) -> bool:
        """Download ERVmap from GitHub or alternative sources."""
        ervmap_dir = self.data_dir / "ervmap"
        if ervmap_dir.exists() and any(ervmap_dir.iterdir()):
            print("✅ ERVmap already downloaded")
            return True
        
        print("📥 Attempting to download ERVmap...")
        print("   Note: Repository may require access or use alternative sources")
        
        # Try multiple potential repository URLs
        repo_urls = [
            'https://github.com/Functional-Genomics/ERVmap.git',
            'https://github.com/mlbendall/ERVmap.git',
        ]
        
        import subprocess
        for url in repo_urls:
            try:
                print(f"   Trying: {url}")
                subprocess.run([
                    'git', 'clone', url, str(ervmap_dir)
                ], check=True, capture_output=True)
                print("✅ ERVmap cloned")
                return True
            except subprocess.CalledProcessError:
                continue
            except FileNotFoundError:
                print("⚠️ Git not found. Please install git or download ERVmap manually.")
                return False
        
        print("⚠️ ERVmap repository not found at standard URLs.")
        print("   Alternative: Download from published paper supplementary data")
        print("   Paper: https://www.nature.com/articles/s41588-019-0371-5")
        print("   Or search: 'ERVmap' on GitHub")
        return False
    
    def download_repeatmasker(self) -> bool:
        """Download RepeatMasker database."""
        rm_dir = self.data_dir / "repeatmasker"
        rm_dir.mkdir(exist_ok=True)
        
        # RepeatMasker libraries
        libraries_url = "https://www.repeatmasker.org/libraries/"
        print("📥 RepeatMasker requires manual download from:")
        print(f"   {libraries_url}")
        print("   Please download and extract to:", rm_dir)
        return False
    
    def download_genbank_erv(self, max_sequences: int = 100, email: Optional[str] = None, force: bool = False) -> bool:
        """Download ERV sequences from GenBank via Entrez."""
        genbank_dir = self.data_dir / "genbank"
        genbank_dir.mkdir(exist_ok=True)
        
        output_file = genbank_dir / "erv_sequences.fasta"
        if output_file.exists() and not force:
            print(f"✅ GenBank ERV sequences already downloaded: {output_file}")
            return True
        elif output_file.exists() and force:
            print(f"🔄 Force re-download: removing existing file...")
            output_file.unlink()
        
        try:
            from Bio import Entrez
            from Bio import SeqIO
            
            # NCBI requires email for Entrez access
            if email:
                Entrez.email = email
            else:
                import os
                Entrez.email = os.getenv("NCBI_EMAIL", "your.email@example.com")
                if Entrez.email == "your.email@example.com":
                    print("⚠️ Set NCBI_EMAIL environment variable or provide --email")
                    print("   NCBI requires email for Entrez API access")
            
            print(f"📥 Searching GenBank for ERV sequences (max {max_sequences})...")
            
            # Search for ERV sequences
            search_term = "endogenous retrovirus[Title] OR ERV[Title]"
            handle = Entrez.esearch(db="nucleotide", term=search_term, retmax=max_sequences)
            record = Entrez.read(handle)
            handle.close()
            
            if not record['IdList']:
                print("⚠️ No ERV sequences found")
                return False
            
            print(f"📊 Found {len(record['IdList'])} ERV sequences")
            print("   Downloading sequences (this may take a while)...")
            
            # Download sequences in batches
            batch_size = 10
            sequences = []
            
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
                
                # Be nice to NCBI servers
                time.sleep(0.5)
            
            # Save to FASTA file
            SeqIO.write(sequences, output_file, "fasta")
            print(f"✅ Saved {len(sequences)} sequences to {output_file}")
            return True
            
        except ImportError:
            print("⚠️ Biopython not installed. Install with: pip install biopython")
            return False
        except Exception as e:
            print(f"❌ GenBank download failed: {e}")
            return False
    
    def download_ensembl_erv(self, species: str = "homo_sapiens") -> bool:
        """Download ERV annotations from Ensembl."""
        ensembl_dir = self.data_dir / "ensembl"
        ensembl_dir.mkdir(exist_ok=True)
        
        output_file = ensembl_dir / f"{species}_erv_annotations.json"
        if output_file.exists():
            print(f"✅ Ensembl ERV annotations already downloaded: {output_file}")
            return True
        
        try:
            import requests
            
            print(f"📥 Downloading Ensembl ERV annotations for {species}...")
            print("   Note: Ensembl may not have direct ERV endpoints")
            print("   This is a placeholder for future API integration")
            
            # Ensembl REST API base URL
            server = "https://rest.ensembl.org"
            
            # Try to get repeat annotations (ERVs are often in repeat databases)
            # Note: This is a simplified example - actual ERV data may require
            # different endpoints or manual database queries
            
            # Example: Get all repeats for a region (would need chromosome/region)
            # For now, we'll create a placeholder structure
            annotations = {
                'species': species,
                'source': 'Ensembl',
                'note': 'Direct ERV endpoints may require custom queries',
                'suggested_approach': [
                    'Use Ensembl Biomart for bulk downloads',
                    'Query RepeatMasker annotations',
                    'Use Ensembl variation API for ERV-associated variants'
                ]
            }
            
            with open(output_file, 'w') as f:
                json.dump(annotations, f, indent=2)
            
            print(f"✅ Created annotation structure at {output_file}")
            print("   For actual ERV data, consider using Ensembl Biomart or RepeatMasker")
            return True
            
        except ImportError:
            print("⚠️ Requests not installed. Install with: pip install requests")
            return False
        except Exception as e:
            print(f"❌ Ensembl download failed: {e}")
            return False
    
    def list_datasets(self):
        """List available datasets."""
        print("\n📋 Available Datasets:\n")
        for key, info in self.datasets.items():
            status = "✅" if (self.data_dir / key).exists() else "⏳"
            print(f"{status} {key}: {info['name']}")
            print(f"   {info['description']}")
            print()
    
    def download_all(self, force: bool = False):
        """Download all available datasets."""
        print("🚀 Starting dataset downloads...\n")
        
        results = {}
        results['ervmap'] = self.download_ervmap()
        results['genbank'] = self.download_genbank_erv()
        results['ensembl'] = self.download_ensembl_erv()
        results['repeatmasker'] = self.download_repeatmasker()
        
        print("\n📊 Download Summary:")
        for name, success in results.items():
            status = "✅" if success else "⏳"
            print(f"{status} {name}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Download biological datasets")
    parser.add_argument('--dataset', choices=['all', 'ervmap', 'genbank', 'ensembl', 'repeatmasker'],
                       default='all', help='Dataset to download')
    parser.add_argument('--list', action='store_true', help='List available datasets')
    parser.add_argument('--force', action='store_true', help='Force re-download')
    parser.add_argument('--email', type=str, help='Email for NCBI Entrez API (required for GenBank)')
    parser.add_argument('--max-sequences', type=int, default=100, 
                       help='Maximum sequences to download from GenBank (default: 100)')
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader()
    
    if args.list:
        downloader.list_datasets()
        return
    
    if args.dataset == 'all':
        downloader.download_all(force=args.force)
    elif args.dataset == 'ervmap':
        downloader.download_ervmap()
    elif args.dataset == 'genbank':
        downloader.download_genbank_erv(max_sequences=args.max_sequences, email=args.email, force=args.force)
    elif args.dataset == 'ensembl':
        downloader.download_ensembl_erv()
    elif args.dataset == 'repeatmasker':
        downloader.download_repeatmasker()


if __name__ == '__main__':
    main()


"""
Sync Validated Data Script

Synchronizes manually validated benchmark data from .data/ to archiX/data/
for publication and archiving. Only syncs datasets that have been verified
to meet CE framework standards.

Usage:
    python3 arXiv/sync.py [--dry-run] [--force] [--dataset DATASET_NAME]
    make agents            # Data synchronization via Makefile
    make sync              # Data synchronization with benchmark dependency

Options:
    --dry-run    Show what would be synced without actually copying
    --force      Skip validation prompts
    --dataset    Sync only specific dataset (scan, cogs, cfq, pcfg, rpm, math)
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import time


class DataSyncManager:
    """Manages synchronization of validated benchmark data."""

    def __init__(self, source_dir: Path = Path("../.data"), target_dir: Path = Path("data")):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.datasets = {
            'scan': 'SCAN (Sequence-to-Sequence with Actions and Navigation)',
            'cogs': 'COGS (Compositional Generalization Semantic Parsing)',
            'cfq': 'CFQ (Compositional Freebase Questions)',
            'pcfg': 'PCFG (Probabilistic Context-Free Grammar)',
            'rpm': 'RPM/RAVEN (Raven\'s Progressive Matrices)',
            'math': 'Math (Mathematical Reasoning Patterns)'
        }

    def get_available_datasets(self) -> Dict[str, bool]:
        """Check which datasets are available in source directory."""
        available = {}
        for dataset_key in self.datasets:
            dataset_path = self.source_dir / dataset_key
            available[dataset_key] = dataset_path.exists() and any(dataset_path.iterdir())
        return available

    def validate_dataset_integrity(self, dataset_key: str) -> bool:
        """Perform basic integrity checks on dataset."""
        dataset_path = self.source_dir / dataset_key

        if not dataset_path.exists():
            print(f"❌ Dataset {dataset_key} not found in {self.source_dir}")
            return False

        # Check for expected files/patterns
        if dataset_key == 'scan':
            expected_files = ['train_split', 'test_split']
            return any((dataset_path / f).exists() for f in expected_files)

        elif dataset_key == 'cogs':
            return (dataset_path / 'data').exists()

        elif dataset_key == 'cfq':
            return (dataset_path / 'cfq').exists()

        elif dataset_key == 'pcfg':
            return (dataset_path / 'cola').exists()

        elif dataset_key == 'rpm':
            return (dataset_path / 'RAVEN-10000').exists()

        elif dataset_key == 'math':
            return (dataset_path / 'SVAMP.json').exists()

        return True

    def sync_dataset(self, dataset_key: str, dry_run: bool = False) -> bool:
        """Sync a specific dataset."""
        source_path = self.source_dir / dataset_key
        target_path = self.target_dir / 'datasets' / dataset_key

        if not source_path.exists():
            print(f"❌ Source dataset {dataset_key} not found")
            return False

        if dry_run:
            print(f"🔍 Would sync {source_path} → {target_path}")
            return True

        try:
            # Create target directory
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy the dataset
            if target_path.exists():
                shutil.rmtree(target_path)

            shutil.copytree(source_path, target_path)

            # Create metadata
            metadata = {
                'dataset_key': dataset_key,
                'dataset_name': self.datasets[dataset_key],
                'synced_at': time.time(),
                'source_path': str(source_path),
                'target_path': str(target_path)
            }

            metadata_file = target_path / '.sync_metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"✅ Synced {dataset_key} to {target_path}")
            return True

        except Exception as e:
            print(f"❌ Failed to sync {dataset_key}: {e}")
            return False

    def sync_benchmark_results(self, dry_run: bool = False) -> bool:
        """Sync benchmark results from .out/ to data/benchmark_results/."""
        source_results = Path("../.out")
        target_results = self.target_dir / 'benchmark_results'

        if not source_results.exists():
            print(f"⚠️ No benchmark results found in {source_results}")
            return False

        if dry_run:
            print(f"🔍 Would sync benchmark results {source_results} → {target_results}")
            return True

        try:
            target_results.mkdir(parents=True, exist_ok=True)

            # Copy JSON result files
            result_files = list(source_results.glob("*.json"))
            for result_file in result_files:
                shutil.copy2(result_file, target_results / result_file.name)
                print(f"✅ Synced {result_file.name}")

            print(f"✅ Synced benchmark results to {target_results}")
            return True

        except Exception as e:
            print(f"❌ Failed to sync benchmark results: {e}")
            return False

    def run_sync(self, dataset: Optional[str] = None, dry_run: bool = False, force: bool = False):
        """Run the complete sync process."""
        print("🔄 Starting data synchronization...")
        print(f"Source: {self.source_dir}")
        print(f"Target: {self.target_dir}")
        print()

        # Check available datasets
        available_datasets = self.get_available_datasets()
        if not available_datasets:
            print("❌ No datasets found in source directory")
            return False

        print("Available datasets:")
        for key, exists in available_datasets.items():
            status = "✅ Available" if exists else "❌ Missing"
            print(f"  {key}: {status}")

        # Determine which datasets to sync
        if dataset:
            if dataset not in available_datasets or not available_datasets[dataset]:
                print(f"❌ Dataset {dataset} not available")
                return False
            datasets_to_sync = [dataset]
        else:
            datasets_to_sync = [k for k, v in available_datasets.items() if v]

        print(f"\nDatasets to sync: {', '.join(datasets_to_sync)}")

        if not force and not dry_run:
            response = input("\nContinue with sync? (y/N): ").lower().strip()
            if response not in ['y', 'yes']:
                print("Sync cancelled")
                return False

        # Validate datasets
        print("\n🔍 Validating datasets...")
        valid_datasets = []
        for dataset_key in datasets_to_sync:
            if self.validate_dataset_integrity(dataset_key):
                valid_datasets.append(dataset_key)
                print(f"✅ {dataset_key} validation passed")
            else:
                print(f"❌ {dataset_key} validation failed")

        if not valid_datasets:
            print("❌ No valid datasets to sync")
            return False

        # Sync datasets
        print("\n📦 Syncing datasets...")
        success_count = 0
        for dataset_key in valid_datasets:
            if self.sync_dataset(dataset_key, dry_run):
                success_count += 1

        # Sync benchmark results
        print("\n📊 Syncing benchmark results...")
        if self.sync_benchmark_results(dry_run):
            print("✅ Benchmark results synced")
        else:
            print("⚠️ Benchmark results sync skipped")

        if dry_run:
            print(f"\n🔍 Dry run complete - would sync {len(valid_datasets)} datasets")
        else:
            print(f"\n✅ Sync complete - {success_count}/{len(valid_datasets)} datasets synced")

        return success_count > 0


def main():
    parser = argparse.ArgumentParser(description="Sync validated benchmark data to arXiv/data/")
    parser.add_argument('--dry-run', action='store_true', help='Show what would be synced without copying')
    parser.add_argument('--force', action='store_true', help='Skip validation prompts')
    parser.add_argument('--dataset', help='Sync only specific dataset (scan, cogs, cfq, pcfg, rpm, math)')

    args = parser.parse_args()

    sync_manager = DataSyncManager()
    success = sync_manager.run_sync(
        dataset=args.dataset,
        dry_run=args.dry_run,
        force=args.force
    )

    exit(0 if success else 1)


if __name__ == '__main__':
    main()


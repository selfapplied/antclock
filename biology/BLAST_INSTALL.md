# Installing BLAST+ for Biology Module

## macOS Installation (Recommended: Homebrew)

### Option 1: Homebrew (Easiest)

```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install BLAST+
brew install blast
```

### Option 2: Direct Download

1. **Download BLAST+**:
   - Go to: https://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Web&PAGE_TYPE=BlastDocs&DOC_TYPE=Download
   - Download the macOS version (`.dmg` file)

2. **Install**:
   - Open the `.dmg` file
   - Drag BLAST+ to Applications or desired location
   - Add to PATH:
     ```bash
     # Add to ~/.zshrc or ~/.bash_profile
     export PATH="/path/to/blast/bin:$PATH"
     ```

### Option 3: Conda (if using conda)

```bash
conda install -c bioconda blast
```

## Verify Installation

After installation, verify BLAST+ is working:

```bash
# Check version
blastn -version

# Should output something like:
# blastn: 2.14.1+
# Package: blast 2.14.1, build Jun 15 2023 14:16:11
```

## Test with Biology Module

Once installed, test the integration:

```bash
# Create BLAST database from GenBank sequences
./run.sh biology/blast/analyze.py --create-db biology/data/genbank/erv_sequences.fasta --db-name genbank_erv_db

# Run BLAST search
./run.sh biology/blast/analyze.py biology/data/genbank/erv_sequences.fasta --db biology/data/blast/genbank_erv_db

# Integrate with ERV analysis
./run.sh biology/erv/integrate_blast.py biology/data/blast/*_blast.txt biology/data/genbank/erv_sequences.fasta
```

## Troubleshooting

### BLAST not found in PATH

If you get "command not found":

```bash
# Find where BLAST was installed
which blastn
# or
find /usr/local -name blastn 2>/dev/null
find /opt -name blastn 2>/dev/null

# Add to PATH in ~/.zshrc:
export PATH="/path/to/blast/bin:$PATH"

# Reload shell
source ~/.zshrc
```

### Permission Issues

If you get permission errors:

```bash
# Make sure BLAST binaries are executable
chmod +x /path/to/blast/bin/*
```

## Alternative: Use Simulated BLAST

If you can't install BLAST+ right now, the module includes a simulator:

```bash
# Generate simulated BLAST results
./run.sh biology/blast/simulate_blast.py sequences.fasta --output blast_results.txt

# Use simulated results for integration
./run.sh biology/erv/integrate_blast.py blast_results.txt sequences.fasta
```

This allows you to test the full pipeline without BLAST+ installed.

## Platform-Specific Notes

### macOS
- **Homebrew**: `brew install blast` (recommended)
- **Direct download**: Use `.dmg` installer from NCBI

### Linux
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install ncbi-blast+

# Or download from NCBI and extract
wget https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/ncbi-blast-*-x64-linux.tar.gz
tar -xzf ncbi-blast-*-x64-linux.tar.gz
export PATH="$PWD/ncbi-blast-*/bin:$PATH"
```

### Windows
1. Download installer from NCBI
2. Run installer
3. Add to PATH or use full path to `blastn.exe`

## Next Steps

Once BLAST+ is installed:

1. ✅ Verify installation: `blastn -version`
2. ✅ Create test database from GenBank sequences
3. ✅ Run real BLAST searches
4. ✅ Compare real vs simulated results
5. ✅ Generate visualizations with real data






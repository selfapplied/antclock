# Biological Research: ERV Analysis with CE Framework

This module applies the CE Tower architecture to understand endogenous retroviruses (ERVs) through causal compositional analysis.

## Connection to Volte Systems

ERVs are modeled as Volte systems (see `arXiv/working.md` Section 5.3):
- **x** = lineage genomic architecture state
- **Q** = species identity / conserved core genes
- **S** = maladaptive load / instability
- **V** = exaptation: viral element → function while preserving lineage identity

## Directory Structure

- `data/` - Downloaded datasets and archives
- `scripts/` - Data download and processing scripts
- `blast/` - BLAST integration and sequence analysis
- `erv/` - ERV-specific analysis modules

## Datasets

### Primary ERV Databases

1. **HERVd** - Human Endogenous Retroviruses Database
   - URL: https://herv.img.cas.cz/
   - Provides: HERV families, integration sites, structural analysis

2. **EnHERV** - Enrichment Analysis of Specific Human Endogenous Retrovirus Patterns
   - Provides: HERV patterns, neighboring genes, enrichment analysis

3. **HERVd Atlas** - HERV-disease associations
   - URL: https://ngdc.cncb.ac.cn/hervd/
   - Provides: Curated HERV-disease links, expression data

4. **ERVmap** - Human ERV annotations and expression
   - GitHub: https://github.com/Functional-Genomics/ERVmap

### Sequence Databases

- **GenBank** - Nucleotide sequences
- **Ensembl** - Genome annotations with ERV data
- **RepeatMasker** - Repeat element identification

## Usage

```bash
# Download datasets
python biology/scripts/download_datasets.py
python biology/scripts/download_datasets.py --dataset genbank --email your@email.com

# Run BLAST analysis
python biology/blast/analyze.py sequences.fasta --create-db sequences.fasta --db-name erv_db

# CE-based ERV analysis with Volte framework
python biology/erv/analyze_erv.py data/erv_sequences.fasta

# Integrate BLAST results with ERV Volte analysis
python biology/erv/integrate_blast.py blast_results.txt query_sequences.fasta
```

## Analysis Pipeline

1. **Download ERV sequences** from GenBank, Ensembl, or other sources
2. **Create BLAST database** from reference sequences
3. **Run BLAST analysis** to identify sequence similarity and conserved regions
4. **Analyze with ERV Volte system** to understand exaptation potential
5. **Integrate results** to connect sequence similarity (CE framework) with evolutionary dynamics (Volte)



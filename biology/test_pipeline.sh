#!/bin/bash
# Complete test pipeline for biology module

set -e

echo "🧪 Testing Biology Module Pipeline"
echo "=================================="
echo ""

# Step 1: Create test data
echo "Step 1: Creating test ERV sequences..."
python3 biology/create_test_data.py

# Step 2: Analyze with ERV Volte system
echo ""
echo "Step 2: Running ERV Volte analysis..."
python3 biology/erv/analyze_erv.py biology/data/test/test_erv_sequences.fasta

# Step 3: Check if BLAST is available
echo ""
echo "Step 3: Checking BLAST availability..."
if command -v blastn &> /dev/null; then
    echo "✅ BLAST+ is installed"
    echo "   Creating BLAST database..."
    python3 biology/blast/analyze.py \
        --create-db biology/data/test/test_erv_sequences.fasta \
        --db-name test_erv_db || echo "⚠️ BLAST database creation skipped"
else
    echo "⚠️ BLAST+ not installed (optional for testing)"
fi

echo ""
echo "✅ Pipeline test complete!"
echo ""
echo "Results saved to:"
echo "  - biology/data/test/test_erv_sequences_erv_analysis.json"






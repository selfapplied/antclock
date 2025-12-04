#!/bin/bash
# AntClock Setup and Run Script
# Sets up environment, installs dependencies, handles font cache issues

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Check if running in correct directory
if [ ! -f "requirements.txt" ]; then
    print_error "requirements.txt not found. Run from antclock directory."
    exit 1
fi

print_info "AntClock Setup and Run Script"
echo "================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d. -f1-2)
print_info "Python version: $PYTHON_VERSION"

# Check if virtual environment exists, create if not
if [ ! -d ".venv" ]; then
    print_info "Creating virtual environment..."
    python3 -m venv .venv
    print_status "Virtual environment created"
else
    print_info "Virtual environment exists"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip (skip if it fails)
print_info "Upgrading pip..."
if pip install --upgrade pip > /dev/null 2>&1; then
    print_status "Pip upgraded"
else
    print_warning "Pip upgrade failed, continuing with existing version"
fi

# Install requirements
print_info "Installing requirements..."
if pip install -r requirements.txt 2>&1; then
    print_status "Requirements installed"
else
    print_warning "Failed to install requirements (network access may be blocked)"
    print_info "Attempting to continue with existing environment..."
fi

# Handle matplotlib font cache issues
print_info "Checking matplotlib font cache access..."

# Test if we can write to home directory for matplotlib
if [ ! -w "$HOME" ] || [ ! -w "$HOME/.matplotlib" ] 2>/dev/null; then
    print_warning "Cannot write to $HOME/.matplotlib (likely read-only filesystem)"
    export MPLCONFIGDIR="/tmp/matplotlib_$(whoami)"
    mkdir -p "$MPLCONFIGDIR"
    print_info "Using MPLCONFIGDIR=$MPLCONFIGDIR"
else
    print_info "Font cache directory is writable"
fi

# Check if custom Python code should be run
if [ "$1" = "--" ]; then
    shift
    print_info "Running custom Python code: $@"
    python3 -c "$@"
    print_status "Custom Python execution complete"
    exit 0
fi

# Determine which script to run based on arguments or default
SCRIPT_TO_RUN=""

if [ $# -eq 0 ]; then
    # Default: check if demo.py exists and is executable
    if [ -f "demo.py" ] && [ -x "demo.py" ]; then
        SCRIPT_TO_RUN="demo.py"
    else
        print_info "No default script specified. Use: $0 <script.py> [args...]"
        print_info "Available scripts:"
        ls -1 *.py 2>/dev/null | head -10
        exit 1
    fi
else
    SCRIPT_TO_RUN="$1"
    shift
fi

# Check if script exists
if [ ! -f "$SCRIPT_TO_RUN" ]; then
    print_error "Script '$SCRIPT_TO_RUN' not found"
    exit 1
fi

# Run the script
print_info "Running $SCRIPT_TO_RUN..."
if python3 "$SCRIPT_TO_RUN" "$@"; then
    print_status "Script executed successfully"

    # Only modify permissions/hashbangs after successful execution

    # Make the script executable if it's not already
    if [ ! -x "$SCRIPT_TO_RUN" ]; then
        chmod +x "$SCRIPT_TO_RUN"
        print_status "Made $SCRIPT_TO_RUN executable"
    fi

    # Check if the script has a proper hashbang
    if [ -f "$SCRIPT_TO_RUN" ] && head -1 "$SCRIPT_TO_RUN" | grep -q "^#!/"; then
        print_info "Script already has hashbang"
    else
        # Add hashbang if missing
        HASHBANG="#!run.sh"
        if ! head -1 "$SCRIPT_TO_RUN" | grep -q "^$HASHBANG"; then
            # Create temp file with hashbang
            TEMP_FILE=$(mktemp)
            echo "$HASHBANG" > "$TEMP_FILE"
            cat "$SCRIPT_TO_RUN" >> "$TEMP_FILE"
            mv "$TEMP_FILE" "$SCRIPT_TO_RUN"
            chmod +x "$SCRIPT_TO_RUN"
            print_status "Added hashbang to $SCRIPT_TO_RUN"
        fi
    fi

    # Make other Python files executable for convenience
    for pyfile in *.py; do
        if [ "$pyfile" != "$SCRIPT_TO_RUN" ] && [ ! -x "$pyfile" ]; then
            chmod +x "$pyfile"
            print_info "Made $pyfile executable"
        fi
    done

    print_status "All setup tasks completed successfully"
else
    print_error "Script execution failed - not modifying permissions"
    exit 1
fi

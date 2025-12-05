#!/bin/bash
# AntClock Setup and Run Script
# Sets up environment, installs dependencies, handles font cache issues
# See README.md for more information.
# Usage:
# ./run.sh <script.py> [args...]
# ./run.sh --dry-run=direct-self <script.py>     # Test Direct Self identity behavior
# ./run.sh --dry-run=proto-self-upgrade <script.py>  # Test Proto-Self upgrade behavior
# ./run.sh --dry-run=foreign-fallback <script.py>    # Test Foreign Identity fallback behavior
# ./run.sh -- <custom_python_code>
# ./run.sh benchmarks/benchmark.py
# ./run.sh demos/antclock.py
# ./run.sh tools/test_types.py

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_header() {
    echo -e "${BLUE}ℹ${NC} $1"
    echo "$1" | sed 's/./=/g'
}

# Check if running in sandboxed environment
is_sandboxed() {
    # Check for common sandbox indicators
    if [ -n "$SANDBOXED" ] || [ -n "$CURSOR_SANDBOX" ] || ! python3 -c "import sys; sys.exit(0)" 2>/dev/null; then
        return 0
    fi
    return 1
}

# Setup Python cache directory
setup_python_cache() {
    export PYTHONPYCACHEPREFIX=".out/pycache"
    print_info "Python cache directory set to $PYTHONPYCACHEPREFIX"

    # Create cache directory if it doesn't exist
    mkdir -p "$PYTHONPYCACHEPREFIX"

    # Clean up any stray __pycache__ directories in source
    find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
}

# Setup virtual environment
setup_virtual_env() {
    print_info "Python version: $(python3 --version | cut -d' ' -f2)"

    if [ -d ".venv" ]; then
        print_info "Virtual environment exists"
        print_info "Activating virtual environment..."
        source .venv/bin/activate
    else
        print_error "Virtual environment not found at .venv"
        print_error "Please create virtual environment: python3 -m venv .venv"
        exit 1
    fi
}

# Setup pip cache directory
setup_pip_cache() {
    export PIP_CACHE_DIR=".pip_cache"
    print_info "Setting up pip cache directory..."
    print_info "Using project-local pip cache: $PIP_CACHE_DIR"

    mkdir -p "$PIP_CACHE_DIR"

    if is_sandboxed; then
        print_warning "Detected sandboxed environment - pip operations may be restricted"
        print_info "Core CE functionality will work, but pip operations may fail"
        print_warning "Sandbox detected - skipping pip upgrade"
        print_warning "Sandbox detected - skipping pip requirements installation"
        print_info "Using existing environment (CE framework works without pip)"
        return
    fi

    # Upgrade pip if not sandboxed
    print_info "Upgrading pip..."
    pip install --upgrade pip --quiet || print_warning "Pip upgrade failed"

    # Install requirements if not sandboxed
    if [ -f "requirements.txt" ]; then
        print_info "Installing requirements..."
        pip install -r requirements.txt --quiet || print_warning "Requirements installation failed"
    else
        print_warning "requirements.txt not found, skipping requirements installation"
    fi
}

# Setup matplotlib
setup_matplotlib() {
    print_info "Checking matplotlib font cache access..."

    # Try to create matplotlib config directory
    if [ -w "$HOME" ] && [ -w "$HOME/.matplotlib" ] 2>/dev/null; then
        export MPLCONFIGDIR="$HOME/.matplotlib"
    else
        print_warning "Cannot write to $HOME/.matplotlib (likely read-only filesystem)"
        export MPLCONFIGDIR="/tmp/matplotlib_$(whoami)"
        mkdir -p "$MPLCONFIGDIR"
        print_info "Using MPLCONFIGDIR=$MPLCONFIGDIR"
    fi

    # Force matplotlib to use non-GUI backend
    export MPLBACKEND=Agg
    print_info "Using MPLBACKEND=$MPLBACKEND (non-GUI backend)"
}

# Identity recognition functions
is_self() {
    local file="$1"
    local first_line
    first_line="$(head -1 "$file" 2>/dev/null || echo "")"
    [[ "$first_line" == "#!run.sh" ]]
}

is_python_file() {
    local file="$1"
    [[ "$file" == *.py ]] || head -1 "$file" 2>/dev/null | grep -q "^#!/.*python"
}

has_main_block() {
    local file="$1"
    grep -q "if __name__ == ['\"']__main__['\"]" "$file" 2>/dev/null
}

is_foreign_identity() {
    local file="$1"
    local first_line
    first_line="$(head -1 "$file" 2>/dev/null || echo "")"
    [[ "$first_line" =~ ^#!/ ]] && [[ "$first_line" != "#!run.sh" ]]
}

# Execute script based on identity
execute_script() {
    local script="$1"
    shift

    if [[ "$script" == *Makefile* ]] || [[ "$script" == *.mk ]]; then
        print_info "Detected Makefile with run.sh hashbang: $script"
        # Check for semantic comment
        if grep -q "^# semantic:" "$script"; then
            local semantic_cmd
            semantic_cmd=$(grep "^# semantic:" "$script" | sed 's/^# semantic: //' | head -1)
            print_info "Executing semantic command: $semantic_cmd"
            eval "$semantic_cmd"
        else
            make -f "$script" run || make -f "$script" all
        fi
        return $?
    fi

    print_info "Running as script: $script"

    # Always set PYTHONPATH to include project root for consistent imports
    export PYTHONPATH="$(pwd):$PYTHONPATH"

    EXEC_CMD="python3 $script"
    $EXEC_CMD "$@"
    local exit_code=$?
    unset PYTHONPATH 2>/dev/null || true  # Clean up
    return $exit_code
}

# Handle script execution with identity recognition
handle_script_execution() {
    local script="$1"
    local script_args="$2"
    local dry_run="$3"
    local expected_behavior="$4"

    # Check if script exists
    if [ ! -f "$script" ]; then
        print_error "Script '$script' not found"
        exit 1
    else
        print_info "Script found: $script"
    fi

    # Determine script directory and name for module execution
    SCRIPT_DIR=$(dirname "$script")
    SCRIPT_NAME=$(basename "$script" .py)

    # Identity-based execution logic
    if is_self "$script"; then
        print_info "Direct Self identity detected (#run.sh)"
        if [ "$dry_run" = true ]; then
            if [ "$expected_behavior" = "direct-self" ]; then
                print_info "DRY RUN: Would execute script and update self-marker on success"
                print_status "DRY RUN: Direct Self identity validated - matches expected behavior"
                exit 0
            else
                print_error "DRY RUN: Expected '$expected_behavior' but detected Direct Self identity"
                exit 1
            fi
        else
            execute_script "$script" "$@"
            local exit_code=$?
            if [ $exit_code -eq 0 ]; then
                # Success: ensure executable and update self-marker
                chmod +x "$script"
                print_status "Made $script executable"
                # Update self-recognition marker (30-day expiration)
                local marker="# AntClock-Self: $(date +%s)"
                if ! grep -q "^# AntClock-Self:" "$script"; then
                    sed -i '' "2i\\
$marker
" "$script"
                else
                    sed -i '' "s/^# AntClock-Self:.*/$marker/" "$script"
                fi
            else
                # Failure: remove hashbang and make non-executable
                sed -i '' '/^#!run.sh$/d' "$script"
                chmod -x "$script"
                print_error "Removed hashbang and executable permission due to failure"
                exit 1
            fi
        fi
    elif is_python_file "$script" && has_main_block "$script"; then
        print_info "Proto-Self detected (Python with main block)"
        if [ "$dry_run" = true ]; then
            if [ "$expected_behavior" = "proto-self-upgrade" ]; then
                print_info "DRY RUN: Would test Proto-Self execution and upgrade to Direct Self on success"
                print_status "DRY RUN: Proto-Self identity validated - would upgrade as expected"
                exit 0
            else
                print_error "DRY RUN: Expected '$expected_behavior' but detected Proto-Self upgrade scenario"
                exit 1
            fi
        else
            # Try running as-is first
            execute_script "$script" "$@"
            local exit_code=$?
            if [ $exit_code -eq 0 ]; then
                print_info "Proto-Self runs successfully as-is"
                return 0
            else
                print_info "Proto-Self failed as-is, trying with run.sh scaffolding..."
                # Try with full environment setup
                setup_environment
                execute_script "$script" "$@"
                local scaffold_exit=$?
                if [ $scaffold_exit -eq 0 ]; then
                    print_info "Proto-Self succeeds with scaffolding - upgrading to Direct Self"
                    # Upgrade to Direct Self
                    sed -i '' '1i\
#!run.sh
' "$script"
                    chmod +x "$script"
                    print_status "Upgraded $script to Direct Self identity"
                    return 0
                else
                    print_info "Proto-Self fails even with scaffolding - not claiming"
                    return $scaffold_exit
                fi
            fi
        fi
    elif is_foreign_identity "$script"; then
        print_info "Foreign identity detected - respecting existing hashbang"
        local first_line
        first_line="$(head -1 "$script")"
        local interpreter="${first_line#\#\!} "
        print_info "Executing with: $interpreter $script"

        if [ "$dry_run" = true ]; then
            # For foreign identity scripts, we expect fallback behavior
            if [ "$expected_behavior" = "foreign-fallback" ]; then
                print_warning "DRY RUN: Interpreter '$interpreter' would fail (not found)"
                print_info "DRY RUN: Would fallback and replace hashbang with '#!run.sh'"
                print_status "DRY RUN: Foreign identity fallback validated - matches expected behavior"
                exit 0
            else
                print_error "DRY RUN: Expected '$expected_behavior' but detected Foreign Identity with fallback potential"
                exit 1
            fi
        else
            # Set PYTHONPATH for consistent imports
            export PYTHONPATH="$(pwd):$PYTHONPATH"
            $interpreter "$script" "$@"
            local exit_code=$?
            unset PYTHONPATH 2>/dev/null || true  # Clean up

            if [ $exit_code -eq 0 ]; then
                return 0
            fi

            # Script failed - check if interpreter exists
            if command -v "$interpreter" >/dev/null 2>&1; then
                print_warning "Script failed with shebang interpreter, trying direct execution..."
                $interpreter "$script" "$@"
                exit_code=$?
                if [ $exit_code -eq 0 ]; then
                    print_info "Direct execution with $interpreter succeeded"
                    return 0
                fi
            else
                print_warning "Interpreter '$interpreter' not found"
            fi

            # Still failed - try replacing with run.sh and running again
            print_info "Attempting fallback to run.sh interpreter..."
            if [[ "$first_line" != "#!run.sh" ]]; then
                sed -i '' "s|^$first_line|#!run.sh|" "$script"
                print_info "Replaced shebang with '#!run.sh', attempting execution..."
            else
                print_info "Script already has '#!run.sh' shebang, retrying execution..."
            fi
            handle_script_execution "$script" "$script_args" "$dry_run" "$expected_behavior"
        fi
        return $?
    else
        print_info "Unclaimed file detected"
        # Check if it's runnable Python
        if is_python_file "$script" && has_main_block "$script"; then
            if [ "$dry_run" = true ]; then
                if [ "$expected_behavior" = "unclaimed-territory" ]; then
                    print_info "DRY RUN: Would test unclaimed Python file with scaffolding"
                    print_status "DRY RUN: Unclaimed territory behavior validated - would claim as Direct Self on success"
                    exit 0
                else
                    print_error "DRY RUN: Expected '$expected_behavior' but detected Unclaimed Territory with Python main block"
                    exit 1
                fi
            else
                print_info "Python file with main block - testing with run.sh scaffolding..."
                setup_environment
            fi
            execute_script "$script" "$@"
            local test_exit=$?
            if [ $test_exit -eq 0 ]; then
                print_info "Unclaimed file succeeds with scaffolding - claiming as Direct Self"
                sed -i '' '1i\
#!run.sh
' "$script"
                chmod +x "$script"
                print_status "Claimed $script as Direct Self"
            else
                print_info "Unclaimed file fails with scaffolding - leaving unclaimed"
                exit 1
            fi
        else
            if [ "$dry_run" = true ]; then
                print_error "DRY RUN: Cannot determine how to execute unclaimed file without Python main block"
                exit 1
            else
                print_error "Cannot determine how to execute: $script"
                exit 1
            fi
        fi
    fi
}

# Main setup function
setup_environment() {
    print_header "AntClock Setup and Run Script"

    setup_python_cache
    setup_virtual_env
    setup_pip_cache
    setup_matplotlib
}

# Main execution logic
main() {
    # Check for dry-run flag with expected behavior
    DRY_RUN=false
    EXPECTED_BEHAVIOR=""
    if [[ "$1" =~ ^--dry-run= ]]; then
        DRY_RUN=true
        EXPECTED_BEHAVIOR="${1#--dry-run=}"
        shift
        print_info "DRY RUN MODE - Expected behavior: '$EXPECTED_BEHAVIOR'"
        print_info "Simulating execution without file modifications"
    fi

    # Check if custom Python code should be run
    if [ "$1" = "--" ]; then
        shift
        if [ "$DRY_RUN" = true ]; then
            print_info "DRY RUN: Would run custom Python code: $@"
            print_status "Custom Python execution simulation complete"
            exit 0
        else
            print_info "Running custom Python code: $@"
            setup_environment
            python3 -c "$@"
            print_status "Custom Python execution complete"
            exit 0
        fi
    fi

    # Determine which script to run based on arguments or default
    SCRIPT_TO_RUN=""

    if [ $# -eq 0 ]; then
        print_info "No script specified. Use: $0 <script.py> [args...]"
        print_info "Available scripts:"
        echo "Core framework: antclock/"
        echo "Demos: demos/"
        echo "Tools: tools/"
        echo "Evaluations: evaluations/"
        echo "Benchmarks: benchmarks/"
        echo ""
        echo "Examples:"
        echo "  $0 demos/demo.py"
        echo "  $0 benchmarks/benchmark.py"
        echo "  $0 evaluations/ce_timing_demo.py"
        exit 1
    else
        SCRIPT_TO_RUN="$1"
        shift  # Remove script name from arguments
        SCRIPT_ARGS="$@"  # Store remaining arguments
    fi

    # Setup environment (skip in dry-run)
    if [ "$DRY_RUN" = false ]; then
        setup_environment
    else
        print_info "DRY RUN: Would setup environment (virtual env, caches, matplotlib)"
    fi

    # Execute script with identity recognition
    handle_script_execution "$SCRIPT_TO_RUN" "$SCRIPT_ARGS" "$DRY_RUN" "$EXPECTED_BEHAVIOR"
}

# Run main function
main "$@"
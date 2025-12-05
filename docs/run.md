# AntClock Run Architecture

## Overview

AntClock implements a sophisticated unified execution system centered around `run.sh` - the universal entry point for all script execution. This architecture unifies agent running with a **viral immunity injection model**, ensuring robust and consistent execution across the entire framework.

## Core Philosophy: Viral Immunity Injection Model

**"Test before trust"** - Like an immune system testing pathogens before allowing cellular integration, scripts undergo rigorous validation before being granted execution permissions in the system's executable namespace.

### Biological Analogy

```python
# Immune System: Test antigens, grant immunity only to safe elements
def immune_response(antigen):
    if test_compatibility(antigen):  # Safe, beneficial
        grant_immunity(antigen)      # Allow propagation
        integrate_into_system()      # Become part of defense
    else:
        reject_antigen()             # Block integration

# run.sh: Test scripts, grant permissions only to working code
def run_sh_execution(script):
    if execute_successfully(script):  # Passes validation
        chmod_executable(script)      # Grant execution rights
        inject_hashbang(script)       # Integrate into ecosystem
        propagate_permissions()       # Enable related scripts
    else:
        deny_permissions()            # Block execution rights
```

This **viral immunity injection model** treats the executable script ecosystem as a living system that carefully vets new code before allowing it to propagate and integrate.

## Architecture Components

### 1. Unified Entry Point: run.sh

`run.sh` serves as the **single source of truth** for script execution, handling:

```bash
# Environment setup
- Virtual environment activation (.venv/)
- Python cache management (PYTHONPYCACHEPREFIX=.out/pycache)
- Matplotlib backend configuration (non-GUI Agg backend)
- Font cache directory setup
- Sandbox environment detection and graceful degradation
- Dependency verification
```

### 2. Viral Immunity: Permissions-Based Testing Model

The system implements a **viral immunity injection model** - treating script execution as biological immunity:

#### Antigen Testing (Script Validation)
```bash
# Scripts undergo rigorous testing before integration
./run.sh some_script.py  # First run - immune system challenge
# If successful: Script proves "non-pathogenic" - grant immunity
# If failed: Script rejected - no execution rights granted
```

#### MHC Self-Recognition Markers
```bash
# Scripts receive "self-recognition" markers after validation
# Marker format: # AntClock-Self: <validation_timestamp>
if script_validated:
    add_self_marker(script, timestamp)  # MHC-like self marker
    grant_execution_rights(script)      # Allow integration into ecosystem
```

#### Self vs Non-Self Discrimination
```bash
# Implemented in run.sh as is_self() function
is_self() {
    # Check if script has valid AntClock self-recognition markers
    local script="$1"
    # Must have hashbang, be executable, and have recent self-marker
    if [ -x "$script" ] && head -1 "$script" 2>/dev/null | grep -q "^#!/.*" && grep -q "^# AntClock-Self:" "$script"; then
        # Check if marker is reasonably recent (within last 30 days)
        local marker_time=$(grep "^# AntClock-Self:" "$script" | head -1 | cut -d: -f2 | tr -d " ")
        local current_time=$(date +%s)
        local age=$((current_time - marker_time))
        [ $age -lt 2592000 ]  # 30 days in seconds
    else
        false
    fi
}

# Execution messages distinguish self vs non-self
if is_self "$SCRIPT_TO_RUN"; then
    print_info "Running self-recognized script: $SCRIPT_TO_RUN"
else
    print_info "Running non-self script: $SCRIPT_TO_RUN"
fi
```

#### Identity Hierarchy & Upgrade Paths
```bash
# Identity recognition hierarchy:
# 1. #!run.sh + executable + self-marker = Full AntClock identity
# 2. #!/usr/bin/env python3 + executable + self-marker = Generic self (upgradable)
# 3. No hashbang + executable + self-marker = New viral infection
# 4. Any markers without self-marker = Non-self (external identity)
```

#### Upgrade Logic for Generic Identities
```bash
# Scripts with generic hashbangs (#/usr/bin/env) are "self" in broader sense
# If they run successfully with run.sh scaffolding, upgrade to viral identity
if script_has_generic_hashbang and runs_with_scaffolding:
    upgrade_to_viral_identity(script)  # Inject #!run.sh
    print_status "Upgraded from generic to viral identity"
```

#### Self-Marker Injection (Immune Memory)
```bash
# After successful validation, ALL scripts receive self-markers
if ! grep -q "^# AntClock-Self:" "$SCRIPT_TO_RUN" 2>/dev/null; then
    echo "# AntClock-Self: $(date +%s)" >> "$SCRIPT_TO_RUN"
    print_status "Added self-recognition marker (immune memory)"
fi
```

#### Example Identity Evolution
```python
# Initial state (generic self)
#!/usr/bin/env python3
print("Generic identity script")

# After viral upgrade
#!run.sh
# AntClock-Self: 1764884549
print("Full AntClock viral identity")
```

Only **self-recognized scripts** integrate fully into the ecosystem, with **upgrade paths** for generic identities that prove compatible with AntClock's viral scaffolding.

### Executable Makefiles: Graph Agents

**Makefiles as First-Class Agents**: The semantic hashbang system extends beyond scripts to create executable build graphs:

```makefile
#!/run.sh
policy: make -f "$SCRIPT_TO_RUN" run || make -f "$SCRIPT_TO_RUN" all

# AntClock Ecosystem Makefile - Executable Graph Agent
run: agents benchmarks timing
	@echo "🎉 AntClock ecosystem activated!"

agents:
	@echo "🔄 Agent synchronization..."

benchmarks:
	@echo "🧬 Benchmark pipeline..."
```

**Multiple Execution Modes**:
```bash
make              # Traditional make usage
./Makefile        # Execute as graph agent via run.sh
make -f Makefile  # Explicit makefile usage
```

**Semantic Policy Declaration**: Makefiles declare their execution policy through `policy:` comments, allowing run.sh to interpret them as runnable computational graphs rather than passive configuration files.

### Advanced Viral Protocol (GPT-51 Model)

The system implements a sophisticated **viral self-recognition protocol** with four identity states:

#### 1. Direct Self (`#!run.sh`)
- **Full AntClock identity**: Scripts with our viral marker
- **Responsibility**: We maintain, upgrade, and revoke these identities
- **Success**: Reinforce markers and permissions
- **Failure**: Revoke identity (remove hashbang, chmod -x)

#### 2. Proto-Self (Generic Python)
- **Generic Python hashbangs**: `#!/usr/bin/env python`, `#!/usr/bin/env python3`
- **Potential**: Can be upgraded to full identity
- **Success**: Upgrade to `#!run.sh` with preserved original hashbang
- **Failure**: Leave unchanged (respect generic Python compatibility)

#### 3. Foreign Identity (Other Interpreters)
- **Non-Python interpreters**: `#!/usr/bin/env bash`, `#!/usr/bin/env node`, etc.
- **Respect**: Never modify foreign identities
- **Behavior**: Run as-is or offer guest execution under run.sh scaffolding

#### 4. Unclaimed Territory
- **No hashbang**: Files without interpreter specification
- **Adoption**: If Python-like with main block, adopt as AntClock identity
- **Success**: Inject `#!run.sh`, make executable, add markers
- **Failure**: Leave untouched

This creates a **viral ecosystem** where identities evolve based on compatibility with AntClock's framework, with careful respect for existing interpreter domains.

This **self-recognition system** ensures the ecosystem can distinguish between **validated AntClock components** ("self") and **external/unvalidated scripts** ("non-self"), preventing autoimmune responses while maintaining ecosystem integrity.

#### Viral Self-Recognition (Identity Injection)
```bash
# Viral propagation based on self-recognition
if script_has_run_sh_hashbang:
    # Already infected - no action needed
    print_info "Script already bears AntClock viral identity"
elif script_has_generic_hashbang:  # e.g., #!/usr/bin/env python3
    # Generic identity - upgrade to viral
    inject_run_sh_hashbang(script)
    print_status "Injected AntClock viral identity (upgrade from generic)"
else:
    # No identity - new infection
    inject_run_sh_hashbang(script)
    print_status "Injected AntClock viral identity (new infection)"
```

#### Viral Cleanup (Failed Integration)
```bash
# On execution failure - remove viral markers
if script_failed:
    if script_has_run_sh_hashbang:
        remove_run_sh_hashbang(script)  # Remove viral identity
        chmod -x script                 # Quarantine (remove permissions)
        print_warning "Removed viral hashbang (failed integration)"
```

The system implements **true viral propagation** - recognizing self, injecting identity markers, and cleaning up failed integrations to maintain ecosystem purity.

### 3. Hashbang Unification System

All executable scripts use `#!/run.sh` as their hashbang:

**Important**: The hashbang is `#!/run.sh` (without the leading `/`). This means scripts will only execute when run from the project root directory where `run.sh` exists. This enforces that all script execution goes through the proper AntClock environment setup.

```bash
# Every executable Python script
head -1 demo.py              # → #!run.sh
head -1 benchmark.py         # → #!run.sh
head -1 tools/test_types.py  # → #!run.sh

# Direct execution routes through run.sh
./demo.py  # → run.sh handles environment + execution
```

#### Automatic Hashbang Injection
```bash
# run.sh adds #!run.sh to scripts without hashbangs
if [ ! -f "$SCRIPT" ] || ! head -1 "$SCRIPT" | grep -q "^#!/"; then
    # Prepend #!run.sh to the script
    echo "#!run.sh" | cat - "$SCRIPT" > temp && mv temp "$SCRIPT"
    chmod +x "$SCRIPT"
fi
```

### 4. Environment Abstraction

`run.sh` abstracts all environment complexity:

#### Development vs Production
```bash
# Handles both pip-enabled and sandboxed environments
if [ "$IN_SANDBOX" = true ]; then
    # Skip pip operations, use existing environment
    print_warning "Sandbox detected - using existing CE framework"
else
    # Full pip install and environment setup
    pip install -r requirements.txt
fi
```

#### Cache Management
```bash
# Centralized cache management
export PYTHONPYCACHEPREFIX=.out/pycache  # Clean source dirs
export MPLCONFIGDIR=/tmp/matplotlib_*    # Handle read-only filesystems
export MPLBACKEND=Agg                     # Non-GUI plotting
```

### 5. Execution Modes

The system supports multiple execution patterns:

#### Direct Script Execution
```bash
./run.sh path/to/script.py [args...]
# → python3 path/to/script.py [args...]
```

#### Module Execution
```bash
./run.sh benchmarks/benchmark.py
# → python3 -m benchmarks.benchmark
```

#### Custom Code Execution
```bash
./run.sh -- "import antclock; print('Hello from AntClock')"
# → python3 -c "import antclock; print('Hello from AntClock')"
```

## Benefits

### 1. **Simplified User Experience**
```bash
# Just execute scripts directly - no environment worries
./demo.py
./benchmarks/benchmark.py
./tools/test_types.py
```

### 2. **Robust Environment Handling**
- Works in sandboxed environments (Cursor, restricted filesystems)
- Handles missing dependencies gracefully
- Manages font cache issues automatically
- Adapts to different Python installations

### 3. **Clean Source Management**
- No `__pycache__/` directories in source code
- Centralized cache in `.out/pycache/`
- Only tested, working scripts become executable
- Automatic permission management

### 4. **Unified Agent Interface**
```bash
# All AntClock components use the same execution model
./zeta_card_interpreter.py    # ζ-card agents
./benchmark.py               # CE framework validation
./demo.py                    # Mathematical demonstrations
./transport_mechanisms.py    # Framework components
```

### 5. **Viral Immunity Protection**
- **Antigen Testing**: Scripts must prove "non-pathogenic" before integration
- **Immune Surveillance**: Failed scripts rejected, preventing "infection spread"
- **Self vs Non-Self Recognition**: Main block detection distinguishes runnable agents from libraries
- **Adaptive Defense**: Successful scripts create immunity patterns for similar code
- **Sterile Environment**: No "viral load" of broken executables polluting the namespace

## Implementation Details

### Permission Propagation
```bash
# After successful execution:
if [ script_succeeded ]; then
    chmod +x "$SCRIPT"                    # Make executable
    add_hashbang_if_missing "$SCRIPT"     # Add #!run.sh
    make_related_scripts_executable       # Grant permissions to siblings
fi
```


### Environment Detection
```bash
# Smart environment detection
IN_SANDBOX=false
if ! python3 -c "import certifi"; then
    IN_SANDBOX=true  # Cursor sandbox detected
fi

DEFAULT_CACHE_WRITABLE=false
if touch "$HOME/.cache/pip_test" 2>/dev/null; then
    DEFAULT_CACHE_WRITABLE=true
fi
```

### Cache Strategy
```bash
# Sophisticated cache management
if [ "$DEFAULT_CACHE_WRITABLE" = false ]; then
    export PIP_CACHE_DIR="$(pwd)/.pip_cache"
fi

export PYTHONPYCACHEPREFIX=.out/pycache
export MPLCONFIGDIR="/tmp/matplotlib_$(whoami)"
```

## Why This Architecture?

### 1. **Agent Unification**
All AntClock components (demos, benchmarks, tools, ζ-cards) share the same execution interface, creating a unified agent ecosystem.

### 2. **Permission Safety**
The "test before trust" model prevents broken or experimental code from becoming part of the system's executable interface.

### 3. **Environment Portability**
Abstracts away environment differences, making AntClock work consistently across development, testing, and deployment environments.

### 4. **User Experience**
Users can focus on mathematics and algorithms, not environment configuration and dependency management.

### 5. **Maintainability**
Centralizes execution logic in one place, making it easier to maintain and extend the system's capabilities.

## Usage Examples

```bash
# Run any component directly
./demos/antclock.py
./benchmarks/ce/synthetic.py
./tools/test_types.py

# Run through run.sh explicitly
./run.sh demos/antclock.py --verbose
./run.sh benchmarks/benchmark.py

# Execute custom code
./run.sh -- "from antclock import clock; print('CE framework ready')"

# All approaches yield the same environment setup
```

## Why Viral Immunity Injection?

### Biological Inspiration
Just as biological immune systems protect organisms by carefully vetting and integrating beneficial elements while rejecting harmful ones, AntClock's execution system protects the codebase ecosystem:

- **Pathogen Rejection**: Broken scripts = viruses that could "infect" the executable namespace
- **Antibody Production**: Hashbangs = immune markers for recognized safe code
- **Herd Immunity**: Related scripts gain rights after ecosystem validation
- **Adaptive Response**: System learns which patterns indicate healthy vs unhealthy code

### System Resilience
This model creates a **self-healing codebase** where:
- Only proven, working code propagates execution rights
- Failed experiments don't create persistent executable clutter
- The system maintains "immune memory" of successful patterns
- New code must earn trust before joining the executable ecosystem

---

**The viral immunity injection model transforms script execution from mere environment management into a living, self-protecting ecosystem where mathematical exploration happens safely within immunological boundaries.**

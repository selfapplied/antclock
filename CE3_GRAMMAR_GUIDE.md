# CE3 Grammar Guide

## Overview

The `.ce3_grammar.json` file defines the **emergent organizational structure** of the AntClock repository. Unlike rigid rules, this grammar establishes **attractor fields** that guide natural repository growth through stigmergic principles—just like how ant colonies self-organize through pheromone trails.

## What is CE3?

**CE3 (Emergent Simplicial Category)** is the third layer of the CE Tower:
- **CE1**: Discrete grammar ([], {}, (), <>)
- **CE2**: Dynamical flow and temporal composition
- **CE3**: Evolutionary restructuring and emergent topology

CE3 recognizes that healthy repositories self-organize. The grammar defines *how* that organization emerges.

## Core Concepts

### 1. Constants (Symbol Tree)
Ground-floor anchors that don't change:
- CE operators: `[]` memory, `{}` domain, `()` transform, `<>` witness
- File extensions, root anchors (LICENSE, README, Makefile)
- Fixed symbols that provide stability

### 2. Attractors
Organizational gravity wells that influence naming and structure:
- **Directory attractors**: `antclock/`, `benchmarks/`, `tests/`, etc.
- **Naming conventions**: snake_case for modules, PascalCase for classes
- **Pheromone strength**: High-traffic paths resist change

### 3. Overflow Rules
What happens when patterns exceed capacity:
- **Output saturation** → Evaporate old outputs
- **Test accumulation** → Consolidate into `.out/`
- **Documentation sprawl** → Migrate to `docs/` or `arXiv/`
- **Naming divergence** → Rename or create new organizational node

### 4. Flow Rules
Self-organizing dynamics:
- **File lifecycle**: Creation → Growth → Stability → Deprecation
- **Directory evolution**: Emergence, consolidation, specialization
- **Stigmergic markers**: High-traffic paths, boundary zones, evaporation targets

### 5. Self-Tag Markers
How files declare their identity:
- Hashbang patterns: `#!run.sh`, `#!/usr/bin/env bash`
- CE annotations: `#@CE1:`, `#@CE2:`, `#@CE3:`
- Metadata headers: `#@ ζ-card`, `#κ: X.XX τ: now ζ: self`

## Repository Topology

The repository self-organizes into concentric rings:

```
Core Attractor: antclock/
    ↓
Validation Ring: benchmarks/, tests/
    ↓
Interface Ring: demos/, tools/, run.sh
    ↓
Documentation Ring: docs/, arXiv/, README.md
    ↓
Ephemeral Cloud: .out/, __pycache__/ (untracked)
```

## Practical Application

### When adding a new file:
1. **Sense** the directory's attractor field
2. **Follow** the naming convention for that domain
3. **Add** appropriate self-tag markers if relevant
4. **Let** the file naturally find its organizational position

### When the repository feels disorganized:
1. **Observe** which overflow rules are triggering
2. **Apply** the suggested response (consolidate, migrate, refactor)
3. **Trust** that simpler organization emerges from following attractor fields

### When patterns conflict:
1. **Recognize** boundary zones (files that bridge domains)
2. **Allow** flexible positioning with dual-attractor influence
3. **Don't force** - let the stronger attractor win naturally

## Growth Patterns

The grammar defines how new structure emerges:
- **CE Tower extension**: New CE layer → `antclock/ce{N}/`
- **Benchmark addition**: New domain → `benchmarks/ce/{domain}.py`
- **Tool creation**: Repeated task → `tools/{purpose}.py`
- **Demo showcase**: Novel application → `demos/{application}/`

## Stigmergic Principles

Like ant colonies:
- **Individual files** follow local attractors (simple rules)
- **Global organization** emerges from interactions (complex structure)
- **Pheromone trails** strengthen frequently-used paths
- **Weak trails evaporate** naturally over time
- **Free will** - contributors make local decisions within attractor fields

## Compositional Coherence

The CE3 grammar is **self-referential**:
- It subjects itself to the patterns it defines
- It evolves slowly, reflecting stable organizational principles
- It provides attractor fields, not rigid constraints
- It acknowledges that healthy systems self-organize from the bottom up

## Acknowledgement

This grammar embodies **ACO (Ant Colony Optimization)** principles:
- Distributed intelligence
- Stigmergic signaling
- Positive feedback loops
- Path continuity through evaporation
- Compositional emergence

The repository, like an ant colony, knows its own structure through the trails we leave. Each file, each commit, each organizational decision deposits pheromones that guide future growth.

**We are enough when we trust the patterns.**

---

## For Maintainers

When reviewing PRs:
1. Check if new files follow attractor patterns
2. Observe if changes strengthen or weaken organizational coherence
3. Trust that minor deviations are acceptable - attractors are guides, not laws
4. Suggest consolidation when overflow rules trigger
5. Celebrate emergence of new organizational nodes when they arise naturally

## For Contributors

When adding code:
1. Read the directory structure as a map of attractor fields
2. Follow the naming conventions for your target domain
3. Add CE annotations if your code participates in the CE Tower
4. Don't overthink - the grammar provides *gravitational fields*, not strict boundaries
5. If something feels wrong organizationally, it probably violates an attractor pattern

---

*"Like ants, we organize not through central command, but through local decisions that respect shared patterns. The colony emerges from the choices of individual workers. So does our codebase."*

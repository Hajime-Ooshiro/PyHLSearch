# Copilot Instructions for HLSearch

## Quick Reference

**Repository Purpose:** Prime number shift search exploring the Hardy-Littlewood conjecture using multiple implementations (NumPy, Numba JIT, bit-packed Python, and beam search).

### Build/Test/Lint Commands

- **Run all tests:** `pytest -q`
- **Run specific test:** `pytest -q tests/test_integration.py::TestIntegration::test_solution_path_validation`
- **Run with verbose output:** `pytest -v`
- **Linting:** No linter configured; Python code follows implicit conventions (docstrings included, type hints used)

### Running the Search Program

```powershell
# Default NumPy implementation
python HLSearch.py --depth 8 --limit 400 --max-depth 249 --target 447

# Quick test (small scale)
python HLSearch.py --depth 3 --cols 50 --limit 0 --output shift_path.txt

# Numba JIT-accelerated version
python HLSearch_Numba.py --numba --depth 8 --limit 400 --max-depth 249 --target 447

# Pure-Python bit-packed version (no NumPy dependency)
python HLSearch_bitpack.py --depth 8 --limit 400 --max-depth 249 --target 447
```

**Common CLI options:**
- `-d, --depth`: Search depth (number of primes to use)
- `-l, --limit`: Pruning threshold (lower bound)
- `--max-depth`: Maximum depth limit
- `-t, --target`: Target value at max depth
- `-p, --primes-count`: Use only first N primes
- `--cols`: Column count (search width)
- `--output`: Output file path for results
- `--checkpoint`: Save checkpoint to JSON file
- `--resume`: Resume from checkpoint JSON
- `--log-level`: Console log level (DEBUG/INFO/WARNING/ERROR)
- `--numba`: Enable Numba JIT (HLSearch_Numba.py only)
- `--cuda`: Prefer CUDA (HLSearch_Numba.py only)

## Architecture & Design

### Four Implementation Variants

1. **HLSearch.py** (Primary)
   - NumPy-based with boolean arrays for bitmasks
   - Optimal for standard workloads
   - State class maintains iterative DFS (no recursion) to avoid call stack limits
   - Supports checkpoint/resume functionality

2. **HLSearch_Numba.py** (Performance)
   - Wraps core bitmask operations in Numba `@njit` with optional `parallel=True`
   - GPU support via `numba.cuda` if available
   - Automatically falls back to NumPy if Numba unavailable
   - Drop-in replacement CLI

3. **HLSearch_bitpack.py** (Lightweight)
   - Pure Python using `int` bitwise operations
   - No NumPy dependency
   - Uses built-in `int.bit_count()` for popcount
   - Identical search logic and CLI to main implementation

4. **HLSearch_Beam.py** (Experimental)
   - Beam search + GPU/CPU parallelization
   - `CPUBackend` for multi-threaded execution
   - Work-in-progress optimization approach

### Core Data Flow

```
SearchConfig (settings)
  ↓
generate_primes(limit) → primes list
  ↓
build_base_rows(primes, cols) → base bool array [len(primes), cols]
  ↓
build_shift_table(primes, cols) → precomputed shift candidates [[p, cols], ...]
  ↓
State(config, shift_table) → iterative DFS explorer
  ↓
state.run(depth) → explores all paths, tracks max_count + shifts
  ↓
Results: max_count (best zero-count), results (count of optimal paths), shifts (path details)
```

### State Class: Iterative DFS Engine

The `State` class replaces recursion with an explicit stack (`_stack`) to avoid Python's recursion limit. This is critical for large `depth` values (up to 249).

**Key attributes:**
- `zero_mask`: Active bitmask at current search depth (AND of row complements)
- `key`: Current path as list of shift indices
- `shift_table`: Pre-computed complemented shift arrays (indexed by [level][shift_value])
- `_stack`: Execution stack storing `[level, base_mask, next_idx, next_p]` frames

**Search invariant:**
- The stack must stay synchronized with `key`: `len(key) == len(stack) - 1`
- Checkpoint/resume must occur only when this invariant holds
- `report_progress()` (which handles checkpointing) is called only after a branching decision is finalized

### Checkpoint/Resume (v1.0.6+)

- Format: JSON with full internal state (`key`, stack frames, bitmasks)
- Resuming guarantees bit-exact reproduction of sequential execution
- Bitpacking: Bitmasks are converted to hex strings to avoid `int` overflow on deserialization

## Key Conventions

### Module Structure
- Monolithic design: single .py file per implementation variant for independence
- Public API in docstrings at module level
- `SearchConfig` (dataclass, frozen) holds all configuration
- Validation happens in `SearchConfig.__post_init__()` at construction time

### Naming Patterns
- `primes`: Sorted list of prime candidates for depth/column selection
- `cols`: Column count (search width/problem dimension)
- `depth`: Actual depth used in current search
- `max_depth`: Upper bound on depth for target-reaching logic
- `target`: Goal threshold at max_depth (stops early if exceeded)
- `limit`: Pruning lower bound (branches with count < limit are abandoned)
- `base_rows`: Raw boolean arrays before complement/shift (shape: [len(primes), cols])
- `shift_table`: Pre-computed complements of shifted rows (indexed [level][shift_idx])
- `zero_mask`: Combined bitmask of all non-zero columns at current depth
- `node_mask`: Intersection of zero_mask and current row complement
- `count`: Popcount of a bitmask (number of non-zero bits)

### Logging & Progress
- Logger: module-level `logger` configured once via `setup_logging()`
- Console level: controlled by `--log-level`; file always DEBUG
- Progress bar: `tqdm` with `mininterval` to reduce update overhead
- Log file: `HLSearch.log` in repo root; rotated at 10MB (keeps 3 backups)

### Testing
- Single test file: `tests/test_integration.py`
- Uses `run_state()` helper to set up, run, and verify a complete search
- `monkeypatch` to override module-level `shift_path_file` for isolation
- Verifies: path validity, bitmask consistency, output format correctness

## Dependency Notes

**Required:**
- Python 3.10+
- `numpy` (for HLSearch.py and HLSearch_Numba.py)
- `tqdm`

**Optional:**
- `numba` (for HLSearch_Numba.py; falls back to NumPy if absent)
- CUDA Toolkit (if using `--cuda` with Numba)

Install all:
```powershell
python -m pip install numpy tqdm numba
```

## Development Tips

### Adding a New Search Variant
- Copy the main module and keep `SearchConfig`, `generate_primes()`, and `State` interface identical
- Modify only the internal representation (e.g., array format, computation kernels)
- Preserve CLI argument parsing and logging setup
- Test with `run_state()` fixture to ensure API compatibility

### Debugging Search Logic
- Use `--log-level DEBUG` to see node expansion trace
- Check `HLSearch.log` for full session history (includes failed branches if needed)
- Verify checkpoint/resume correctness: run `--checkpoint`, interrupt, then `--resume` and compare outputs

### Performance Profiling
- Use `timeit` on `build_shift_table()` and `state.run()` separately
- Numba: warm up with a small run first (JIT compilation overhead)
- bit-packed variant: compare `int.bit_count()` performance across Python versions

### Large-Scale Runs
- Checkpoint frequently (`--checkpoint` with short `--checkpoint-interval` if working with huge depths)
- Monitor memory: `State` stores `shift_table` (size ≈ sum of primes × cols bytes)
- For `depth > 100`, prefer bit-packed or Numba variants for speed and memory efficiency

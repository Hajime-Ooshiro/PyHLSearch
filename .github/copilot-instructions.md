# Copilot Instructions for HLSearch

## Project Scope

`HLSearch.py` is the tracked, standard implementation: a NumPy-based search program for prime shifts related to the Hardy-Littlewood second conjecture. Treat `bk/` as ignored local experimental/archive material, not as a supported implementation or a source of required compatibility.

## Commands

Run commands from the repository root.

```powershell
# Entire collected test suite
pytest -q

# A single test
pytest -q tests/test_search.py::test_search_uses_prime_ranges_as_shift_candidates

# A single integration test
pytest -q tests/test_integration.py::TestIntegration::test_solution_path_validation

# Small CLI search suitable for a smoke run
python HLSearch.py --depth 3 --cols 50 --output shift_path.txt

# Standard search
python HLSearch.py --depth 8 --max-depth 249 --target 447
```

There is no configured linter or build step. The project requires Python 3.10+, NumPy, and tqdm. CuPy is optional: `State` uses it only for popcount when the CLI is invoked with `--cuda` and a CUDA device is available; otherwise it uses NumPy.

## Architecture

The application is intentionally monolithic. `SearchConfig` owns validated search parameters and defaults, `generate_primes()` creates the ordered candidate primes, and `build_base_rows()` creates a Boolean matrix where each row represents one prime. `build_shift_table()` packs the precomputed complemented candidates into `uint64` words consumed by the search: `shift_table[level][shift]` has shape `(ceil(cols / 64),)`, and each level's table has shape `(prime, ceil(cols / 64))`.

`State` performs an iterative depth-first search over shift choices. It intersects a parent `zero_mask` with one precomputed row complement to produce `node_mask`, counts active entries, and prunes branches below `max_count`. At leaves, it records paths whose count equals `target` and final maximum-count paths; `shifts` is their de-duplicated union, `target_results` counts target matches, and `results` counts final maximum-count paths. The CLI creates the config and table, runs the state, then writes the result file as `max_count`, `results`, `target_results`, and one shift list per line.

Checkpointing serializes the entire resumable DFS state to version-2 JSON: settings, `key`, masks, result aggregates, node count, and stack frames. Boolean masks are encoded through `np.packbits` into Python integers so widths above 64 bits round-trip safely. A resumed search rejects checkpoints whose settings do not exactly match the active search.

## Repository Conventions

- Keep the public module surface in `HLSearch.py`: `SearchConfig`, `State`, `generate_primes`, `build_base_rows`, `build_shift_table`, `shift_array`, `setup_logging`, and `parse_args`. Add validation for configuration or table-shape contracts at construction time, following the existing `ValueError` messages.
- Preserve packed `uint64` NumPy masks throughout the search implementation. `build_shift_table()` stores complements up front, so the hot search loop must use `base_mask & shift_table[level][i]` rather than recomputing shifts or complements.
- The DFS stack uses mutable frames in the exact form `[level, base_mask, next_idx, next_p]`. `key` and `_stack` must remain synchronized: while an active branch is represented, `len(key) == len(_stack) - 1`. Save a checkpoint only after resolving the current branch and restoring or extending this relationship.
- Keep checkpoint files atomic: write the JSON to `<checkpoint>.tmp`, then replace the target. Maintain version `2` and its exact `settings` compatibility check when changing the saved state.
- `State.run()` always closes its tqdm progress bar and writes a final checkpoint when configured. Preserve that cleanup behavior when adjusting search control flow.
- Tests import the root `HLSearch` module directly. Use small prime lists and column counts for deterministic tests, build tables with the same primes and `cols` as `SearchConfig`, and use `tmp_path` for output/checkpoint files.
- `pytest.ini` restricts collection to `tests/` and excludes `bk/`. Do not rely on code under `bk/` for test coverage.

## Runtime Behavior

Use `--checkpoint path.json` to save periodic state and `--resume path.json` to resume it with identical parameters. The CLI log is `HLSearch.log` beside `HLSearch.py`; file logging is always DEBUG while `--log-level` controls console output. Search results are written to `--output` (default: `shift_path.txt`).

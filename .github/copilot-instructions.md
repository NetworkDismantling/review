# Copilot Instructions — Network Dismantling Framework

## Project Overview

Library for the *"Robustness and resilience of complex networks"* Nature Reviews Physics paper.
Integrates 14+ dismantling algorithms (GND, CoreHD, CI, GDM, EGND, EI, FINDER, Min-Sum, VE, NE, heuristics…) behind a **unified Python interface**, with C++ backends compiled via CMake.
Published for the research community: code quality, clarity, and correctness are paramount.

## Architecture

```
dismantler.py (CLI)  →  @dismantling_method registry (auto-discovered)
                     →  common/dismantlers.py (threshold/lcc strategies)
                     →  common/external_dismantlers/ (C++ .so, CMake)
                     →  greedy_reinsertion/ (shared reinsertion module, CMake)
                     →  storage/pandas/parquet.py (ParquetDataFrameWriter)
```

- **Each algorithm lives in its own folder** (e.g., `network_dismantling/GND/`) with a `python_interface.py` written for interoperability. C++ algorithms are invoked via `subprocess` or compiled extensions.
- **Auto-discovery**: `network_dismantling/__init__.py` uses `pkgutil.walk_packages` to import all modules ending with `*.python_interface` or `*.reinsertion_interface`, triggering `@dismantling_method` / `@reinsertion_method` decorators that populate the global `dismantling_methods` and `reinsertion_methods` dicts.
- **Registries**: `dismantling_methods` and `reinsertion_methods` in `__init__.py`. Each entry stores metadata (name, short_name, method_type, citation, etc.) in a `DismantlingMethod` or `ReinsertionMethod` instance respectively. Both classes are callable (delegate to their `.function`).
- **`method_type`**: optional string on `DismantlingMethod` — e.g. `"heuristic"` for node-metric heuristics, `None` for ML/optimisation algorithms. Used for filtering and categorisation.
- **`@dismantler_wrapper`** (in `common/dismantlers.py`) transforms a scoring function into a full orchestrator: calls the predictor, runs the dismantler strategy, computes AUC (via `scipy.integrate.simpson`), and returns a result dict.
- **Heuristics integration**: node-metric heuristics (degree, eigenvector centrality, PageRank, betweenness, random) are registered via `heuristics/python_interface.py` using `@dismantling_method(method_type="heuristic")` + `@dismantler_wrapper`. The scoring functions live in `heuristics/sorters.py`.
- **Graphs**: always `graph_tool.Graph`, undirected, with `vertex_properties["static_id"]` (int, original index) and `graph_properties["filename"]`. Directed graphs are auto-converted with a warning.
- **Removal**: frozen `dataclass` with slots (`removal_num`, `node_id`, `prediction`, `lcc_size`, `slcc_size`). Values are **absolute counts**, not fractions.
- **Storage (Parquet)**: `removals` column uses `list<struct<…>>` (efficient Arrow format, NOT JSON strings). Compression: Snappy. Supports cross-session append. **Always use `ParquetDataFrameWriter` as context manager** (`with ParquetDataFrameWriter(...) as writer:`) — NOT the deprecated `start_df_writer`, and NOT manual `.start()` / `.close()`.
- **Reinsertion**: `greedy_reinsertion/` is the canonical reinsertion package. It provides `reverse_greedy_reinsertion()` with two interchangeable back-ends: a **graph-tool C++ extension** (`libreinsertion_gt.so`, preferred — in-process, no temp files) and a **subprocess binary** (`reinsertion`, fallback). The public API auto-selects the best available back-end. All algorithms (GDM, CoreGDM, GND+R, multiscale, vertex) import from here.

## Environment & Setup

- **Pixi** manages the environment (conda deps including `graph-tool`, Boost, CMake).
- PyTorch/PyG installed dynamically via `pixi run setup-pytorch` (auto-detects CUDA).
- Local package: `pixi run install-local` (editable `pip install -e .`, requires activated env for `CONDA_PREFIX`).
- All-in-one: `pixi run setup-all`.
- Tests: `pixi run test` (pytest).

## Development Conventions

- **Language**: English in all code, docstrings, comments, and commit messages.
- **Indentation**: 4 spaces. Named parameters for functions with >2 params.
- **Delegate to libraries**: use well-enstablished libraries instead of reimplementing functionality. For example, use `argparse` for CLI parsing, `pathlib` for path handling, `graph_tool` for graph operations, `numpy`/`scipy` for numerical work, `pandas`/`pyarrow` for data storage. Check `common/`, existing code, and available libraries before writing new utilities. For graph operations, prefer built-in `graph_tool` methods (e.g., `graph.to_undirected()`) over manual implementations (e.g., filtering edges).
- **Parametric code**: avoid hardcoded values for features, layer sizes, thresholds, relationship types. Use arguments or configuration.
- **Modular, low-coupling design**: small functions with single responsibility. Follow software engineering best practices (SRP, DRY, low coupling).
- **Prefer simple solutions**: use state-of-the-art techniques only when simpler approaches are demonstrably insufficient.
- **No monkeypatching**: avoid modifying imported modules or classes at runtime. Use subclassing or composition instead.
- **Code from notebooks → .py**: validated notebook code gets consolidated into `.py` modules for performance and clarity.
- **Avoid mutating shared state**: do not modify lists/dicts passed by reference from callers (e.g., `expected_columns += [...]` mutates the caller's list). Use `.copy()` or create new lists.
- **pandas ≥ 2.0 required**: use `pd.concat([df, new])` instead of the removed `df.append()`.
- **When uncertain on architectural or data decisions, ask for clarification before implementing** — do not guess on design choices that affect correctness or interoperability.
- **Information integrity**: never fabricate numbers, benchmark results, or code behavior. Every quantitative claim must come from the actual code, data, or an explicit user statement. If unsure, flag with a placeholder and ask.
- **Deprecation decorator**: use `@deprecated` from `network_dismantling.common.deprecation` (Python 3.13+ native with fallback for older versions).
- **Logging**: use the `logging` module for all output, never `print()`. For subprocesses, use `LogPipe` to capture stdout/stderr.
- **File placement**: new modules should be placed in the most specific existing package (e.g., `common/`, `utils/`) or a new subpackage if warranted. Avoid creating new top-level packages unless necessary. Also avoid placing general utilities in non-general packages (e.g., don't put a general CSV→Parquet converter in `GDM/`). Chose carefully between `common/` (for code shared across multiple algorithms) and `utils/` (for standalone scripts/utilities that are not imported by the main codebase).

## Testing

- Framework: **pytest**, test files in `tests/`.
- Critical features require **extensive tests** by default.
- Use `tempfile.TemporaryDirectory` for file-based tests; clear C++ cache (`cpp_cache.clear()`) in `autouse` fixtures.
- Test naming: `test_<module>_<behavior>.py`. See `tests/test_dismantler_comparison.py` for the pattern of comparing Python vs C++ dismantlers on varied graph topologies (mid-sized, disconnected, complete, star, random).

## Storage Migration (In Progress)

- **CSV → Parquet**: primary storage is now Parquet (`common/storage/pandas/parquet.py`). CSV reader exists as legacy (`common/storage/pandas/csv.py`).
- Migration utilities in `network_dismantling/utils/`: `migrate_to_parquet.py` for converting old CSV files to Parquet.
- DataFrame columns: `network`, `network_size`, `heuristic`, `slcc_peak_at`, `lcc_size_at_peak`, `slcc_size_at_peak`, `removals`, `static`, `r_auc`, `rem_num`, `prediction_time`, `dismantle_time`, `threshold`.
- Removals schema: `list<struct<removal_num:uint32, id:uint32, prediction:float32, lcc_size:uint32, slcc_size:uint32>>`.
- **`idx` and `file` columns**: synthetic, added at read-time by `read_without_columns()` for internal tracking. They must NOT be written to Parquet or leaked into the output schema. Always drop them before writing.
- **Old format compatibility**: legacy CSV stored `lcc_size / network_size` (fraction) instead of absolute `lcc_size`. The `network_size` column was stored separately. New format stores absolute counts; a mapping `network → network_size` is needed for old-format conversion.

## C++ Integration Patterns

- **Subprocess pattern** (CoreHD, CI, EI, EGND, Decycler): writes graph to temp file → invokes compiled binary → reads result from temp file. Compilation triggered on-the-fly via CMake (or `make` for legacy algorithms). Use `logging` module and `LogPipe` (never `print()`) for subprocess output. For temporary files, use `tempfile.NamedTemporaryFile` with `delete=False` (Windows compatibility) and ensure cleanup.
- **Compiled extension — dismantler** (`common/external_dismantlers/`): CMake → `dismantler.so`, exposes `thresholdDismantler`, `lccThresholdDismantler`, and `Graph` class to Python. Uses a global cache with `deepCopy()`.
- **Compiled extension — reinsertion** (`greedy_reinsertion/`): CMake → `libreinsertion_gt.so` (graph-tool Boost.Python module) + `reinsertion` subprocess binary. Two targets in one `CMakeLists.txt`. The graph-tool extension operates in-process on `graph_tool.Graph` objects; the binary is the fallback.
- **Migration target**: subprocess-based C++ algorithms should be migrated to Boost.Python compiled extensions (like `external_dismantlers/` or `greedy_reinsertion/`) for better performance. Requires extensive testing — behavior must not change (unless fixing bugs).
- **Node indexing**: Python is 0-indexed; many C++ extensions use 1-indexed — watch for `+1`/`-1` at boundaries.
- Setup hooks (`_setup_hook.py`) let each submodule define its own build step, executed during `pip install -e .`.
- `graph-tool` integration: the libraries is written in C++ and exposes a Python interface via Boost.Python. For C++ extensions that use `graph-tool`, the recommended approach is to compile them as Boost.Python modules that can be imported directly in Python (like `libreinsertion_gt.so`), rather than using subprocesses. This allows direct manipulation of `graph_tool.Graph` objects without serialization overhead. If subprocesses are necessary (e.g., for legacy code), ensure that the graph is written to disk in a format that preserves all necessary properties (e.g., using `graph_tool.save_graph()`) and read back correctly.

## Known Issues & Workarounds

- **OpenMP on macOS**: PyTorch import can fail due to OpenMP conflict with NumPy. Workaround: `import numpy` before `import torch` (see `setup_pytorch.py` L146–149).
- **`OMP_NUM_THREADS`**: currently commented out in `dismantler.py` L80 — may need re-enabling for reproducibility on multi-core systems.
- **Multiprocessing**: uses `spawn` start method (required for CUDA compatibility). Uses `deadpool.Deadpool` if available, else `ProcessPoolExecutor`.
- **FINDER**: requires TensorFlow 1.15 (no longer available in modern environments). Integration pending; may require rewrite to TF2/PyTorch.
- **Reinsertion subprocess sort bug**: the `reinsertion` binary's `sort_nodes_by_degree` function has off-by-one errors (`degree(i+1, g)` and `W[nodes[i]-1]`) because the BGL graph with `vecS` is 0-indexed. Affects sort ORDER only (strategy ≠ 0), not the SET of selected nodes. The graph-tool C++ extension (`libreinsertion_gt.so`) does NOT have this bug.

## Future Directions

- **Bond percolation (edge dismantling)**: current framework supports site percolation (node removal) only. `edge_dismantling/` (CoreHS, eGDM, heuristics) is WIP for bond percolation support. `common/from_node_to_edge.py` provides the transformation decorator.
- **C++ extension migration**: replace subprocess-based C++ integrations with Boost.Python compiled modules for in-process execution, eliminating temp-file I/O overhead. Requires per-algorithm regression tests against the subprocess version.
- **Parquet migration completion**: replace all remaining `start_df_writer` calls with `ParquetDataFrameWriter` context manager. Fix plot/table scripts for Parquet compatibility.
- **Old storage format conversion**: build a converter for legacy CSV files where LCC was stored as fraction (`lcc_size / network_size`) to new absolute-count format.
- **FINDER modernization**: port from TensorFlow 1.15 to a supported framework.
- **Reinsertion subprocess bug fix**: fix the `sort_nodes_by_degree` off-by-one in `greedy_reinsertion/reinsertion.cpp` (use `degree(i, g)` and `W[nodes[i]]`). Requires regression tests.
- **Legacy reinsertion cleanup**: the old per-algorithm `reinsertion/` folders (GDM, multiscale, vertex) and their `reinsert.py` wrappers are now dead code — all algorithms import from `greedy_reinsertion/`. These folders can be removed once confirmed unused.

## Documentation Policy

- Maintain **one primary doc file** (this `copilot-instructions.md` for AI agents; `README.md` for users, `CHANGES.md` as changelog). Do NOT create additional markdown files unless targeting a distinct audience.
- Update this file when discovering new patterns, conventions, workarounds, issues or useful information during development. This is the single source of truth for AI agents, along with the code itself, user statements and (if any) user conventions, preferences and instructions.
- For user-facing documentation (e.g., `README.md`, docstrings), maintain a clear distinction between user-facing content (how to use the library, explanations of algorithms, etc.) and developer-facing content (implementation details, architectural decisions, etc.). The former goes in `README.md` and docstrings; the latter goes in `copilot-instructions.md`. Avoid mixing the two audiences in the same document.

## Key File Reference

| Purpose | Path |
|---|---|
| CLI entry point | `network_dismantling/dismantler.py` |
| Algorithm registry & auto-discovery | `network_dismantling/__init__.py` |
| Decorator definitions | `network_dismantling/_sorters.py` |
| Dismantler strategies + wrapper | `network_dismantling/common/dismantlers.py` |
| Removal dataclass | `network_dismantling/common/removal.py` |
| Parquet storage | `network_dismantling/common/storage/pandas/parquet.py` |
| Graph loading | `network_dismantling/common/loaders.py`, `common/dataset_providers.py` |
| C++ fast dismantler | `network_dismantling/common/external_dismantlers/` |
| Greedy reinsertion (canonical) | `network_dismantling/greedy_reinsertion/` |
| Reinsertion backward-compat shim | `network_dismantling/common/reinsertion/` |
| Heuristics (auto-registered) | `network_dismantling/heuristics/python_interface.py` |
| Heuristic scoring functions | `network_dismantling/heuristics/sorters.py` |
| Node→edge transform | `network_dismantling/common/from_node_to_edge.py` |
| Deprecation decorator | `network_dismantling/common/deprecation.py` |
| Migration tool (CSV→Parquet) | `network_dismantling/utils/migrate_to_parquet.py` |
| Dataset files | `dataset/` |
| Tests | `tests/` |
| Plot/table scripts | `network_dismantling/plot.py`, `table_output*.py` |

# Copilot Instructions — Network Dismantling Framework

## Project Overview

Library for the *"Robustness and resilience of complex networks"* Nature Reviews Physics paper.
Integrates 14+ dismantling algorithms (GND, CoreHD, CI, GDM, EGND, EI, FINDER, Min-Sum, VE, NE, heuristics…) behind a **unified Python interface**, with C++ backends compiled via Makefile/CMake.
Published for the research community: code quality, clarity, and correctness are paramount.

## Architecture

```
dismantler.py (CLI)  →  @dismantling_method registry (auto-discovered)
                     →  common/dismantlers.py (threshold/lcc strategies)
                     →  common/external_dismantlers/ (C++ .so, CMake)
                     →  storage/pandas/parquet.py (ParquetDataFrameWriter)
```

- **Each algorithm lives in its own folder** (e.g., `network_dismantling/GND/`) with a `python_interface.py` written for interoperability. C++ algorithms are invoked via `subprocess` or compiled extensions.
- **Auto-discovery**: `network_dismantling/__init__.py` uses `pkgutil.walk_packages` to import all `*.python_interface` modules, triggering `@dismantling_method` decorators that populate the global `dismantling_methods` dict.
- **`@dismantler_wrapper`** (in `common/dismantlers.py`) transforms a scoring function into a full orchestrator: calls the predictor, runs the dismantler strategy, computes AUC (via `scipy.integrate.simpson`), and returns a result dict.
- **Graphs**: always `graph_tool.Graph`, undirected, with `vertex_properties["static_id"]` (int, original index) and `graph_properties["filename"]`. Directed graphs are auto-converted with a warning.
- **Removal**: frozen `dataclass` with slots (`removal_num`, `node_id`, `prediction`, `lcc_size`, `slcc_size`). Values are **absolute counts**, not fractions.
- **Storage (Parquet)**: `removals` column uses `list<struct<…>>` (efficient Arrow format, NOT JSON strings). Compression: Snappy. Supports cross-session append.

## Environment & Setup

- **Pixi** manages the environment (conda deps including `graph-tool`, Boost, CMake).
- PyTorch/PyG installed dynamically via `pixi run setup-pytorch` (auto-detects CUDA).
- Local package: `pixi run install-local` (editable `pip install -e .`, requires activated env for `CONDA_PREFIX`).
- All-in-one: `pixi run setup-all`.
- Tests: `pixi run test` (pytest).

## Development Conventions

- **Language**: English in all code, docstrings, comments, and commit messages.
- **Indentation**: 4 spaces. Named parameters for functions with >2 params.
- **Delegate to libraries**: use `graph_tool`, `numpy`, `scipy`, `pandas`, `pyarrow` APIs instead of reimplementing (e.g., use `graph.to_undirected()`, not manual edge filtering). Always check `common/`,  libraries (already available first, to avoid too many dependencies) and existing code before writing new utilities.
- **Parametric code**: avoid hardcoded values for features, layer sizes, thresholds, relationship types. Use arguments or configuration.
- **Modular, low-coupling design**: small functions with single responsibility. Follow software engineering best practices (SRP, DRY, low coupling, ...).
- **Prefer simple solutions**: use state-of-the-art techniques only when simpler approaches are demonstrably insufficient.
- **No monkeypatch**: for instance, avoid `unittest.mock.patch` of production code internals; prefer dependency injection or test-specific fixtures.
- **Code from notebooks → .py**: validated notebook code gets consolidated into `.py` modules for performance and clarity.

## Testing

- Framework: **pytest**, test files in `tests/`.
- Critical features require **extensive tests** by default.
- Use `tempfile.TemporaryDirectory` for file-based tests; clear C++ cache (`cpp_cache.clear()`) in `autouse` fixtures.
- Test naming: `test_<module>_<behavior>.py`. See `tests/test_dismantler_comparison.py` for the pattern of comparing Python vs C++ dismantlers on varied graph topologies (mid-sized, disconnected, complete, star, random).

## Storage Migration (In Progress)

- **CSV → Parquet**: primary storage is now Parquet (`common/storage/pandas/parquet.py`). CSV reader exists as legacy (`common/storage/pandas/csv.py`).
- Migration utilities in `network_dismantling/utils/`: `migrate_csv_to_parquet.py`, `convert_csv_to_parquet.py`, `verify_parquet_schema.py`.
- DataFrame columns: `network`, `network_size`, `heuristic`, `slcc_peak_at`, `lcc_size_at_peak`, `slcc_size_at_peak`, `removals`, `static`, `r_auc`, `rem_num`, `prediction_time`, `dismantle_time`, `threshold`. Removals is `list<struct<removal_num:int, node_id:int, prediction:float, lcc_size:int, slcc_size:int>>`. This format allows efficient storage and querying with PyArrow, without string parsing overhead. Future changes should be made with backward compatibility in mind.

## C++ Integration Patterns

- **Subprocess pattern** (GND, CoreHD, CI, EI, EGND, Decycler): writes graph to temp file → invokes compiled binary → reads result from temp file. Compilation triggered on-the-fly via `make` in the algorithm's folder.
- **Compiled extension** (`common/external_dismantlers/`): CMake → `dismantler.so`, exposes `thresholdDismantler`, `lccThresholdDismantler`, and `Graph` class to Python. Uses a global cache with `deepCopy()`.
- **Node indexing**: Python is 0-indexed; many C++ extensions use 1-indexed — watch for `+1`/`-1` at boundaries.
- Setup hooks (`_setup_hook.py`) let each submodule define its own build step, executed during `pip install -e .`.

## Known Issues & Workarounds

- **OpenMP on macOS**: PyTorch import can fail due to OpenMP conflict with NumPy. Workaround: `import numpy` before `import torch` (see `setup_pytorch.py` L146–149).
- **`OMP_NUM_THREADS`**: currently commented out in `dismantler.py` L80 — may need re-enabling for reproducibility on multi-core systems.
- **Multiprocessing**: uses `spawn` start method (required for CUDA compatibility). Uses `deadpool.Deadpool` if available, else `ProcessPoolExecutor`.

## Documentation Policy

- Maintain **one primary doc file** (this `copilot-instructions.md` for AI agents; `README.md` for users, `CHANGES.md` as changelog). Do NOT create additional markdown files unless targeting a distinct audience.
- Update this file when discovering new patterns, conventions, or workarounds.

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
| Node→edge transform | `network_dismantling/common/from_node_to_edge.py` |
| Dataset files | `dataset/` |
| Tests | `tests/` |

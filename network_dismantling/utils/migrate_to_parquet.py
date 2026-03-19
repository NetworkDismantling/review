#!/usr/bin/env python3
"""Unified CSV → Parquet migration tool for dismantling results.

Converts one or more CSV dismantling result files to Parquet format with:

- Proper ``list<struct>`` schema for the ``removals`` column.
- Optional conversion of legacy fraction-based LCC/SLCC sizes to absolute counts
  (needed for CSV files produced before the absolute-count storage change).
- Maximum-compression gzip backup of each input CSV before deletion.
- Robust post-write validation (row count, columns, non-removals content).
- Optional merge of multiple CSVs into a single Parquet file — makes downstream
  reads more efficient thanks to better column compression over larger row groups.
- Per-run journal log capturing all warnings and errors.

Usage examples::

    # Convert a single file (output: results.parquet)
    python migrate_to_parquet.py -f results.csv

    # Convert several files individually
    python migrate_to_parquet.py -f a.csv b.csv c.csv

    # Merge several files into one Parquet
    python migrate_to_parquet.py -f a.csv b.csv --merge -o merged.parquet

    # Recursively convert all CSVs in a folder
    python migrate_to_parquet.py --folder out/df/

    # Convert with legacy fraction → absolute-count conversion for removals
    python migrate_to_parquet.py -f old_results.csv --convert-removals
"""

import gzip
import logging
import shutil
import sys
from argparse import ArgumentParser, Namespace
from ast import literal_eval
from pathlib import Path
from typing import List, Optional

import pandas as pd

from network_dismantling.common.logging.tqdm_logging_handler import TqdmLoggingHandler
from network_dismantling.common.storage.pandas.csv import df_reader as csv_df_reader
from network_dismantling.common.storage.pandas.parquet import (
    df_reader as parquet_df_reader,
    ParquetDataFrameWriter,
    verify_removals_schema,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interaction helpers
# ---------------------------------------------------------------------------

def _confirm(prompt: str, args: Namespace) -> bool:
    """Ask a yes/no question.  Returns ``True`` immediately if ``--yes`` is set."""
    if getattr(args, "yes", False):
        logger.info(f"[--yes] {prompt}")
        return True

    try:
        answer = input(f"\n{prompt} [y/N]: ").strip().lower()
        return answer in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def _resolve_network_sizes(
    csv_files: List[Path],
    args: Namespace,
) -> Optional[dict]:
    """Build a ``{network_stem: num_vertices}`` mapping by scanning graph folders.

    Called only when ``--convert-removals`` is active and the CSV lacks a
    ``network_size`` column.  Interactive: shows the discovered name → path
    mapping and asks the user to confirm before loading any graphs.

    Args:
        csv_files: Input CSV paths (used to collect unique network names).
        args: Parsed CLI arguments (must include ``args.networks``).

    Returns:
        Dict mapping each network stem to its vertex count, or ``None`` if the
        user aborted or required networks could not be matched unambiguously.
    """
    from collections import defaultdict

    from network_dismantling.common.dataset_providers import list_files as list_network_files
    from network_dismantling.common.loaders import load_graph

    if not args.networks:
        logger.error(
            "--convert-removals: the CSV is missing the 'network_size' column. "
            "Provide graph folder(s) with --networks to look up sizes automatically."
        )
        return None

    # 1. Collect unique network names from all input CSVs (cheap — skip removals)
    network_names: set = set()

    for csv_file in csv_files:
        try:
            df_meta = csv_df_reader(
                files=csv_file,
                include_removals=False,
                raise_on_missing_file=True,
                logger=logger,
            )
            if "network" not in df_meta.columns:
                logger.warning(
                    f"{csv_file.name}: no 'network' column — skipping for size lookup"
                )
                continue

            network_names.update(df_meta["network"].astype(str).unique())

        except Exception as exc:
            logger.error(
                f"Could not read {csv_file.name} to extract network names: {exc}"
            )
            return None

    if not network_names:
        logger.error("No network names found in the input CSVs")
        return None

    logger.info(f"Found {len(network_names)} unique network name(s) across all CSVs")

    # 2. Scan provided folders for graph files (.gt or .graphml)
    network_folders = [Path(p).resolve() for p in args.networks]

    all_graph_files: List[Path] = []
    for folder in network_folders:
        if not folder.is_dir():
            logger.warning(f"--networks: {folder} is not a directory — skipping")
            continue

        try:
            found = list_network_files(location=folder, filter="*")
            all_graph_files.extend(found)
        except FileNotFoundError:
            logger.warning(f"--networks: no graph files found in {folder}")

    if not all_graph_files:
        logger.error("No graph files found in the provided --networks folder(s)")
        return None

    # 3. Group all graph paths by stem to detect duplicates
    stem_to_paths: dict = defaultdict(list)
    for path in all_graph_files:
        stem_to_paths[path.stem].append(path)

    # 4. Classify each network name: matched / missing / ambiguous
    matched:   dict = {}            # name → single Path
    missing:   List[str] = []
    ambiguous: dict = {}            # name → [Path, ...]

    for name in sorted(network_names):
        candidates = stem_to_paths.get(name, [])
        if not candidates:
            missing.append(name)
        elif len(candidates) > 1:
            ambiguous[name] = candidates
        else:
            matched[name] = candidates[0]

    # 5. Print the full mapping for the user to review
    sep = "=" * 72
    print(f"\n{sep}")
    print("  Network size lookup — name → graph file")
    print(sep)

    for name in sorted(matched):
        print(f"  OK   {name:<40s}  {matched[name]}")

    if missing:
        print()
        for name in missing:
            print(f"  ✗    {name:<40s}  NOT FOUND")

    if ambiguous:
        print()
        for name, paths in sorted(ambiguous.items()):
            print(f"  ?    {name:<40s}  AMBIGUOUS ({len(paths)} matches):")
            for p in paths:
                print(f"         {p}")

    print(sep)

    if missing or ambiguous:
        issues = len(missing) + len(ambiguous)
        logger.error(
            f"{issues} network(s) could not be matched unambiguously. "
            "Adjust --networks so that each network stem appears exactly once."
        )
        return None

    # 6. Ask user to confirm the name → path mapping
    if not _confirm(
        f"Is the above name → path mapping correct? ({len(matched)} networks)", args
    ):
        logger.info("Aborted by user.")
        return None

    # 7. Load each graph and count vertices
    logger.info("Loading graphs to measure vertex counts…")

    network_sizes: dict = {}

    for name, path in sorted(matched.items()):
        try:
            g = load_graph(str(path))
            n = g.num_vertices()
            network_sizes[name] = n
            logger.debug(f"  {name}: {n:,} vertices")
        except Exception as exc:
            logger.error(f"Failed to load {path}: {exc}")
            return None

    # 8. Show resulting sizes and ask for final confirmation before modifying anything
    print(f"\n{sep}")
    print("  Loaded network sizes")
    print(sep)

    for name, n in sorted(network_sizes.items()):
        print(f"  {name:<40s}  {n:>10,} nodes")

    print(sep)

    if not _confirm(
        f"Proceed with fraction → absolute conversion using these {len(network_sizes)} sizes?",
        args,
    ):
        logger.info("Aborted by user.")
        return None

    return network_sizes


# ---------------------------------------------------------------------------
# Removals helpers  (only needed by the --convert-removals path)
# ---------------------------------------------------------------------------

def _parse_removals_column(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the ``removals`` column from its CSV string repr to a list of tuples.

    CSV files store removals as ``"[(num, id, pred, lcc, slcc), ...]"`` strings.
    Parsing is required before ``_convert_removals_fractions_to_absolute`` so that
    the fraction-to-absolute arithmetic can operate on individual numeric fields.

    When writing via ``ParquetDataFrameWriter``, parsing is deferred automatically
    (``convert_removals_to_struct`` inside the writer handles the string form).
    """
    if "removals" not in df.columns:
        return df

    def _parse(val):
        if val is None or (isinstance(val, str) and not val):
            return None
        if isinstance(val, str):
            try:
                return literal_eval(val)
            except (ValueError, SyntaxError) as exc:
                logger.warning(f"Could not parse removals value: {exc}")
                return None

        return val  # already a list / tuple

    df = df.copy()
    df["removals"] = df["removals"].apply(_parse)

    return df


def _convert_removals_fractions_to_absolute(df: pd.DataFrame) -> pd.DataFrame:
    """Convert LCC/SLCC sizes in removals from fractions ``[0, 1]`` to absolute counts.

    Required for legacy CSV files where ``lcc_size`` was stored as
    ``lcc_count / network_size``.  Requires a ``network_size`` column.
    The ``removals`` column must already be parsed (list of tuples, not a string).

    Args:
        df: DataFrame with parsed ``removals`` and a ``network_size`` column.

    Returns:
        New DataFrame with removals converted to absolute counts.

    Raises:
        ValueError: If the ``network_size`` column is missing.
    """
    if "removals" not in df.columns:
        return df

    if "network_size" not in df.columns:
        raise ValueError(
            "--convert-removals requires a 'network_size' column in the CSV"
        )

    logger.info(f"Converting {len(df)} rows: removals fraction → absolute counts")

    def _convert_row(row):
        removals = row["removals"]
        if not removals:
            return removals

        n = int(row["network_size"])

        return [
            (
                int(r[0]),
                int(r[1]),
                float(r[2]),
                int(round(float(r[3]) * n)),   # lcc_size: fraction → absolute
                int(round(float(r[4]) * n)),   # slcc_size: fraction → absolute
            )
            for r in removals
        ]

    df = df.copy()
    df["removals"] = df.apply(_convert_row, axis=1)

    return df


# ---------------------------------------------------------------------------
# Backup / cleanup helpers
# ---------------------------------------------------------------------------

def _backup_csv(csv_file: Path) -> Optional[Path]:
    """Compress ``csv_file`` to ``<same_name>.csv.gz`` at maximum gzip level (9).

    Returns the backup path on success, ``None`` on failure.
    """
    backup = csv_file.with_suffix(".csv.gz")

    try:
        with (
            open(csv_file, "rb") as src,
            gzip.open(str(backup), "wb", compresslevel=9) as dst,
        ):
            shutil.copyfileobj(src, dst)

        orig_kb = csv_file.stat().st_size / 1024
        bkp_kb  = backup.stat().st_size / 1024
        ratio   = csv_file.stat().st_size / backup.stat().st_size

        logger.info(
            f"Backed up {csv_file.name} → {backup.name}  "
            f"({orig_kb:.1f} KB → {bkp_kb:.1f} KB, {ratio:.1f}× compression)"
        )
        return backup

    except OSError as exc:
        logger.error(f"Failed to create backup for {csv_file}: {exc}")
        return None


def _remove_csv(csv_file: Path) -> None:
    """Delete the original CSV after a successful backup."""
    try:
        csv_file.unlink()
        logger.info(f"Removed original {csv_file.name}")
    except OSError as exc:
        logger.error(f"Could not remove {csv_file}: {exc}")


# ---------------------------------------------------------------------------
# Core read / pre-process step
# ---------------------------------------------------------------------------

def _read_and_prepare(csv_file: Path, args: Namespace) -> Optional[pd.DataFrame]:
    """Read a CSV file and apply all requested preprocessing.

    Uses the shared CSV reader, which handles column typing, multi-file
    concatenation, and deduplication.  The ``idx`` and ``file`` synthetic
    columns added by the reader are dropped before returning so they are never
    written to Parquet.

    The ``removals`` column is left as-is for the normal write path —
    ``ParquetDataFrameWriter`` / ``convert_removals_to_struct`` handle the
    CSV string form internally.  Only with ``--convert-removals`` does the
    column need to be parsed explicitly, to apply the fraction → absolute math.

    Returns:
        Prepared DataFrame, or ``None`` on failure.
    """
    try:
        df = csv_df_reader(
            files=csv_file,
            include_removals=True,
            raise_on_missing_file=True,
            at_least_one_file=True,
            logger=logger,
        )
        logger.info(
            f"Read {csv_file.name}: {len(df)} rows × {len(df.columns)} columns"
        )

    except Exception as exc:
        logger.error(f"Failed to read {csv_file}: {exc}")
        return None

    # Drop synthetic columns — must not be written to Parquet
    df = df.drop(columns=["idx", "file"], errors="ignore")

    # Legacy conversion: parse removals strings, then multiply sizes by network_size
    if args.convert_removals:
        # If network_size is absent, inject it from the pre-built size map
        # (built interactively by _resolve_network_sizes before any files are modified)
        if "network_size" not in df.columns:
            size_map = getattr(args, "network_size_map", None) or {}

            df = df.copy()
            df["network_size"] = df["network"].astype(str).map(size_map)

            missing_nets = df["network_size"].isna()
            if missing_nets.any():
                logger.error(
                    f"{csv_file.name}: no size found for network(s): "
                    f"{sorted(df.loc[missing_nets, 'network'].astype(str).unique())}"
                )
                return None

            logger.info(
                f"{csv_file.name}: injected 'network_size' from loaded graph files "
                f"(column was absent in the CSV)"
            )

        df = _parse_removals_column(df)

        try:
            df = _convert_removals_fractions_to_absolute(df)
        except Exception as exc:
            logger.error(f"Removals conversion failed for {csv_file.name}: {exc}")
            return None

    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(original_df: pd.DataFrame, parquet_file: Path) -> bool:
    """Validate that the written Parquet faithfully reproduces the original DataFrame.

    Checks:
    1. File exists and is non-empty on disk.
    2. Parquet can be read back via the canonical reader.
    3. Row count matches.
    4. Non-``removals`` column set matches.
    5. Non-``removals`` values match (dtype-flexible comparison).

    ``removals`` is excluded from the content comparison because its in-memory
    representation after struct round-trip differs from the input
    (Python list-of-tuples vs. list-of-dicts / numpy scalars).

    Returns:
        ``True`` if all checks pass.
    """
    if not parquet_file.exists() or parquet_file.stat().st_size == 0:
        logger.error("Parquet file missing or empty after write")
        return False

    try:
        parquet_df = parquet_df_reader(
            files=parquet_file,
            include_removals=False,
            raise_on_missing_file=True,
            logger=logger,
        )
        # Drop synthetic columns added by the reader
        parquet_df = parquet_df.drop(columns=["idx", "file"], errors="ignore")

    except Exception as exc:
        logger.error(f"Cannot read back Parquet file: {exc}")
        return False

    # Row count
    if parquet_df.shape[0] != original_df.shape[0]:
        logger.error(
            f"Row count mismatch: "
            f"Parquet {parquet_df.shape[0]} vs CSV {original_df.shape[0]}"
        )
        return False

    # Column set (Parquet is read without removals; compare only the non-removals columns)
    expected_cols = {c for c in original_df.columns if c != "removals"}
    if set(parquet_df.columns) != expected_cols:
        logger.error(
            f"Column mismatch:\n"
            f"  Parquet: {sorted(parquet_df.columns)}\n"
            f"  CSV:     {sorted(expected_cols)}"
        )
        return False

    # Content comparison (dtype-flexible).
    # Normalize backing types first: both readers may return the same logical values
    # in different containers (e.g. pandas Categorical vs ArrowStringArray for "network").
    # Convert all such columns to plain object dtype so assert_frame_equal only checks values.
    check_cols = sorted(expected_cols)

    def _to_objects(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in out.columns:
            dtype_str = str(out[col].dtype)
            if dtype_str == "category" or "string" in dtype_str or "arrow" in dtype_str.lower():
                out[col] = out[col].astype(object)
        return out

    try:
        pd.testing.assert_frame_equal(
            _to_objects(parquet_df[check_cols]).reset_index(drop=True),
            _to_objects(original_df[check_cols]).reset_index(drop=True),
            check_dtype=False,
            check_names=True,
        )
    except AssertionError as exc:
        logger.error(f"Data content mismatch: {exc}")
        return False

    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert_single(
    csv_file: Path,
    output_file: Optional[Path],
    args: Namespace,
) -> bool:
    """Convert a single CSV file to Parquet.

    Pipeline:
    1. Read CSV → DataFrame (via shared CSV reader).
    2. Parse / convert removals (only if ``--convert-removals``).
    3. Write Parquet (via ``ParquetDataFrameWriter``; handles struct conversion).
    4. Validate row count, columns, and non-removals data.
    5. Create ``.csv.gz`` backup at maximum gzip compression.
    6. Remove original CSV.

    Args:
        csv_file: Source CSV path.
        output_file: Destination Parquet path; defaults to ``<input>.parquet``.
        args: Parsed CLI arguments.

    Returns:
        ``True`` on full success (Parquet written, CSV backed up and removed).
    """
    csv_file = csv_file.resolve()
    output_file = (output_file or csv_file.with_suffix(".parquet")).resolve()

    if output_file.exists():
        logger.warning(
            f"Output {output_file.name} already exists — skipping {csv_file.name}"
        )
        return False

    df = _read_and_prepare(csv_file, args)
    if df is None:
        return False

    # Write via the canonical Parquet writer — handles struct conversion for removals
    with ParquetDataFrameWriter(
        output_file=output_file,
        columns=df.columns.tolist(),
        logger=logger,
    ) as writer:
        writer.write(df)

    # Schema sanity-check: ensure removals landed as list<struct>
    verify_removals_schema(output_file, logger)

    if not _validate(df, output_file):
        logger.error(
            "Validation failed — removing Parquet, preserving original CSV"
        )
        output_file.unlink()
        return False

    # Backup, then remove the original CSV
    backup = _backup_csv(csv_file)
    if backup is None:
        logger.error(
            "Backup failed — Parquet is valid but original CSV was NOT removed. "
            "Check disk space and permissions."
        )
        # Parquet is fine; surface the backup failure but do not abort
        return True

    _remove_csv(csv_file)
    return True


def merge_files(
    csv_files: List[Path],
    output_file: Path,
    args: Namespace,
) -> bool:
    """Merge multiple CSV files into a single Parquet file.

    Merging into one file yields better Snappy compression — column compression
    operates within a row group, so more rows → better ratios — and fewer
    file-open round trips for downstream reads.

    Pipeline:
    1. Read all CSVs → DataFrames (via shared CSV reader).
    2. Validate that all files share the same column set.
    3. Concatenate and write merged Parquet (via ``ParquetDataFrameWriter``).
    4. Validate merged Parquet.
    5. Back up and remove all original CSVs.

    Args:
        csv_files: Source CSV paths.
        output_file: Destination Parquet path.
        args: Parsed CLI arguments.

    Returns:
        ``True`` on full success.
    """
    output_file = output_file.resolve()

    if output_file.exists():
        logger.warning(f"Output {output_file} already exists — aborting merge")
        return False

    logger.info(f"Merging {len(csv_files)} CSV files → {output_file.name}")

    frames: List[pd.DataFrame] = []
    for csv_file in csv_files:
        df = _read_and_prepare(csv_file, args)
        if df is None:
            logger.error(f"Aborting merge — read failure on {csv_file.name}")
            return False
        frames.append(df)

    # All files must share the same column set (extra/missing columns → silent data loss)
    all_cols = [frozenset(f.columns) for f in frames]
    if len(set(all_cols)) > 1:
        logger.error("CSV files have incompatible columns — cannot merge safely:")
        for csv_file, cols in zip(csv_files, all_cols):
            logger.error(f"  {csv_file.name}: {sorted(cols)}")
        return False

    merged = pd.concat(frames, ignore_index=True)
    logger.info(f"Total after merge: {len(merged):,} rows")

    # Write via the canonical Parquet writer
    with ParquetDataFrameWriter(
        output_file=output_file,
        columns=merged.columns.tolist(),
        logger=logger,
    ) as writer:
        writer.write(merged)

    verify_removals_schema(output_file, logger)

    if not _validate(merged, output_file):
        logger.error(
            "Merged Parquet validation failed — "
            "removing Parquet, preserving all CSVs"
        )
        output_file.unlink()
        return False

    # Back up and remove each original CSV
    all_backed_up = True
    for csv_file in csv_files:
        backup = _backup_csv(csv_file)
        if backup is None:
            all_backed_up = False
        else:
            _remove_csv(csv_file)

    if not all_backed_up:
        logger.warning("Some CSV backups failed — check logs above")

    return True


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _add_journal_handler(log: logging.Logger, journal: Path) -> None:
    """Attach a file handler that records warnings and errors to a journal file."""
    try:
        handler = logging.FileHandler(str(journal), mode="a")
        handler.setLevel(logging.WARNING)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)-8s - %(message)s")
        )
        log.addHandler(handler)
        log.info(f"Warnings/errors also logged to {journal}")

    except OSError as exc:
        log.warning(f"Could not open journal file {journal}: {exc}")


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Unified CSV → Parquet migration tool for dismantling results",
        epilog=__doc__,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-f", "--files",
        nargs="+",
        type=Path,
        help="CSV file(s) to convert.",
    )
    input_group.add_argument(
        "-F", "--folder",
        type=Path,
        help="Recursively convert all CSVs in this folder.",
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help=(
            "Output Parquet path.  For single-file conversion defaults to "
            "<input>.parquet.  Required with --merge."
        ),
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help=(
            "Merge all input CSVs into a single Parquet file. "
            "Requires --output. More efficient than individual files."
        ),
    )
    parser.add_argument(
        "--convert-removals",
        action="store_true",
        dest="convert_removals",
        help=(
            "Convert LCC/SLCC sizes in the removals column from fractions [0,1] "
            "to absolute node counts (for legacy CSV files)."
        ),
    )
    parser.add_argument(
        "--pattern",
        default="*.csv",
        help="Glob pattern for --folder mode (default: %(default)s).",
    )
    parser.add_argument(
        "--networks",
        nargs="+",
        type=Path,
        default=None,
        metavar="FOLDER",
        help=(
            "Folder(s) containing the original graph files (.gt / .graphml). "
            "Required by --convert-removals when the CSV lacks a 'network_size' column: "
            "vertex counts are looked up from the actual graph files and injected into "
            "the DataFrame before the fraction → absolute conversion."
        ),
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip all interactive confirmation prompts (non-interactive mode).",
    )

    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s :: %(levelname)-8s :: %(message)s",
        handlers=[TqdmLoggingHandler()],
    )

    parser = _build_parser()
    args = parser.parse_args()

    if args.merge and args.output is None:
        parser.error("--merge requires --output")

    # ---- collect input files -----------------------------------------------

    if args.folder is not None:
        folder = Path(args.folder).resolve()
        if not folder.is_dir():
            logger.error(f"Not a directory: {folder}")
            sys.exit(1)

        _add_journal_handler(logger, folder / "migrate_to_parquet.log")

        csv_files = sorted(
            f
            for f in folder.rglob(args.pattern)
            if f.is_file() and not f.name.endswith(".csv.gz")
        )
        logger.info(f"Found {len(csv_files)} CSV file(s) in {folder}:")

    else:
        csv_files = [Path(f).resolve() for f in args.files]
        _add_journal_handler(logger, csv_files[0].parent / "migrate_to_parquet.log")

    if not csv_files:
        logger.warning("No CSV files to process.")
        sys.exit(0)

    logger.info(f"Found the following {len(csv_files)} CSV file(s) to convert:")
    # for f in csv_files:
    #     logger.info(f"\t- {f}")

    # ---- network size lookup -----------------------------------------------
    # When --convert-removals is requested but the CSV has no 'network_size' column,
    # we load the actual graph files and inject the column before conversion.
    # This is done upfront — before any file is modified — so the user can review
    # and confirm the name → path mapping interactively.

    args.network_size_map = None

    if args.convert_removals:
        try:
            _peek = csv_df_reader(
                files=csv_files[0],
                include_removals=False,
                raise_on_missing_file=True,
                logger=logging.getLogger("dummy"),
            )
            _needs_size_lookup = "network_size" not in _peek.columns
        except Exception:
            _needs_size_lookup = False

        if _needs_size_lookup:
            args.network_size_map = _resolve_network_sizes(csv_files, args)
            if args.network_size_map is None:
                sys.exit(1)

    # ---- confirm before making any modifications ---------------------------

    n_files = len(csv_files)
    if args.merge:
        if n_files == 1:
            logger.warning("Only one input file — --merge has no effect")
        elif n_files > 1 and args.output is None:
            logger.error("--merge requires --output when merging multiple files")
            sys.exit(1)
        action = f"merge {n_files} CSV file(s) into {args.output.name}"
    else:
        action = f"convert {n_files} CSV file(s) to Parquet"

    if not _confirm(f"Proceed: {action}?", args):
        logger.info("Aborted by user.")
        sys.exit(0)

    # ---- run ---------------------------------------------------------------

    if args.merge:
        success = merge_files(csv_files, args.output, args)

    elif len(csv_files) == 1:
        success = convert_single(csv_files[0], args.output, args)

    else:
        results = [convert_single(f, None, args) for f in csv_files]
        n_ok = sum(results)
        success = all(results)
        logger.info(f"Converted {n_ok}/{len(csv_files)} files successfully")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

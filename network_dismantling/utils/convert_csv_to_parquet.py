import logging
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from network_dismantling.common.logging.tqdm_logging_handler import TqdmLoggingHandler

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(TqdmLoggingHandler())

    parser = ArgumentParser(description="")

    parser.add_argument(
        "-f",
        "--folder",
        type=Path,
        default=None,
        required=True,
        nargs="+",
        help="Output DataFrame file(s) location",
    )

    args = parser.parse_args()

    for folder in args.folder:

        folder = Path(folder).resolve()

        if not folder.exists() or not folder.is_dir():
            logger.error(f"Folder {folder} does not exist or is not a directory. Skipping.")
            continue

        journal_file = folder / "convert_csv_to_parquet.log"
        while journal_file.exists():
            journal_file = journal_file.with_name(journal_file.name.replace(".log", "_1.log"))

        logger.info(f"Writing errors and warnings to {journal_file}")
        journal_handler = logging.FileHandler(journal_file, mode='w')
        journal_handler.setLevel(logging.WARNING)
        journal_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(journal_handler)

        logger.info(f"Searching for CSV files in {folder}...")

        # Find all CSV files in the folder and its subfolders
        #  Freeze the list of files to avoid modifying it during iteration by creating the backup folder
        list_of_files = list(folder.rglob("*.csv"))
        for file in list_of_files:
            if "csv_backup" in str(file):
                continue
                
            backup_folder = file.parent / "csv_backup"
            if not backup_folder.exists():
                backup_folder.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created backup folder: {backup_folder}")

            backup_file = backup_folder / file.name

            if backup_file.exists():
                logger.warning(f"Backup file {backup_file} already exists. Skipping backup for {file}.")
                continue

            # file = Path(file)
            if not file.is_file():
                # logger.warning(f"File {file} does not exist or is not a file. Skipping.")
                continue

            output_file = file.with_suffix(".parquet")
            if output_file.exists():
                logger.warning(f"Output file {output_file} already exists. "
                               f"Remove it or rename the input file manually. "
                               f"Skipping conversion for {file}."
                               )
                continue

            logger.info(f"Converting {file} to Parquet. "
                        f"Output file will be {output_file}")

            try:
                # Step 1: Read CSV
                logger.debug(f"Reading CSV file {file}...")
                df = pd.read_csv(str(file))
                logger.info(f"Read {len(df)} rows, {len(df.columns)} columns from {file}")

                # Step 2: Write Parquet
                logger.debug(f"Writing Parquet file {output_file}...")
                df.to_parquet(str(output_file),
                              index=False,
                              engine="pyarrow",
                              compression="snappy",
                              )

                # Step 3: Validate conversion
                logger.debug(f"Validating conversion...")
                parquet_df = pd.read_parquet(str(output_file), engine="pyarrow")

                validation_passed = True
                
                if parquet_df.shape != df.shape:
                    logger.error(f"Shape mismatch: Parquet {parquet_df.shape} vs CSV {df.shape}")
                    validation_passed = False
                
                if list(parquet_df.columns) != list(df.columns):
                    logger.error(f"Column mismatch: Parquet {list(parquet_df.columns)} vs CSV {list(df.columns)}")
                    validation_passed = False
                
                # Check values (handling NaN comparisons)
                if not parquet_df.equals(df):
                    # Try comparing with NaN tolerance
                    try:
                        pd.testing.assert_frame_equal(parquet_df, df, check_dtype=False)
                        logger.debug(f"DataFrames match (with dtype flexibility)")
                    except AssertionError as e:
                        logger.error(f"DataFrame content mismatch: {e}")
                        validation_passed = False
                
                if validation_passed:
                    # Step 4: Move original to backup (only after successful validation)
                    logger.info(f"Moving original CSV to backup: {backup_file}")
                    file.rename(backup_file)
                    
                    # Log conversion statistics
                    logger.info(f"✓ Successfully converted {file.name}")
                    logger.info(f"  CSV size: {backup_file.stat().st_size / 1024:.1f} KB")
                    logger.info(f"  Parquet size: {output_file.stat().st_size / 1024:.1f} KB")
                    compression_ratio = backup_file.stat().st_size / output_file.stat().st_size
                    logger.info(f"  Compression: {compression_ratio:.2f}x smaller")
                else:
                    logger.error(f"✗ Validation failed for {output_file}")
                    logger.error(f"  Removing invalid Parquet file...")
                    if output_file.exists():
                        output_file.unlink()
                    logger.error(f"  Original CSV preserved at {file}")
            
            except Exception as e:
                logger.exception(f"Error during conversion of {file}: {e}")
                # Clean up partial Parquet file if it exists
                if output_file.exists():
                    logger.info(f"Cleaning up partial Parquet file {output_file}")
                    try:
                        output_file.unlink()
                    except Exception as cleanup_error:
                        logger.error(f"Failed to clean up {output_file}: {cleanup_error}")


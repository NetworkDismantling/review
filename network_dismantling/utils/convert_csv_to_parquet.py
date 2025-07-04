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

            df = pd.read_csv(str(file),
                             )

            df.to_parquet(str(output_file),
                          index=False,
                          engine="fastparquet",
                          compression="snappy",
                          )

            parquet_df = pd.read_parquet(str(output_file))

            if not parquet_df.equals(df):
                logger.error(f"DataFrame read from {output_file} does not match the original DataFrame from {file}.")
            else:
                logger.info(f"Successfully converted {file} to Parquet format at {output_file}.")

            # # Optionally, compress the original CSV file as a backup
            # compressed_file = file.with_suffix(".csv.gz")
            # if not compressed_file.exists():
            #     logger.info(f"Compressing original CSV file {file} to {compressed_file}.")
            #     df.to_csv(str(compressed_file), index=False, compression='gzip')
            # else:
            #     logger.warning(f"Compressed file {compressed_file} already exists. Skipping compression.")
            #
            # # Check if the compressed file was created successfully
            # if compressed_file.exists():
            #     df_compressed = pd.read_csv(compressed_file)
            #     if not df_compressed.equals(df):
            #         logger.error(f"DataFrame read from {compressed_file} does not match the original DataFrame from {file}.")
            #     else:
            #         logger.info(f"Compressed file created successfully: {compressed_file}.")
            # else:
            #     logger.error(f"Failed to create compressed file: {compressed_file}.")

            # Move the original CSV file to the backup folder
            try:
                file.rename(backup_file)
                logger.info(f"Moved original CSV file {file} to backup folder {backup_folder}.")
            except Exception as e:
                logger.error(f"Failed to move {file} to {backup_file}: {e}")


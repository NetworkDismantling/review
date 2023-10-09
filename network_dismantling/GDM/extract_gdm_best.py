#   This file is part of GDM (Graph Dismantling with Machine learning),
#   proposed in the paper "Machine learning dismantling and
#   early-warning signals of disintegration in complex systems"
#   by M. Grassia, M. De Domenico and G. Mangioni.
#
#   GDM is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   GDM is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with GDM.  If not, see <http://www.gnu.org/licenses/>.

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from network_dismantling.common.df_helpers import df_reader, get_df_columns
from network_dismantling.common.helpers import extend_filename


def load_and_clean_df(args):
    if args.output_file is None:
        args.output_file = extend_filename(args.file, "_best_runs")

    if args.output_file != args.file:
        raise RuntimeError("Output file is the same as input file.")

    print(f"Storing to {args.output_file}")

    df = df_reader(args.file,
                   include_removals=True,
                   )

    kwargs = {
        "path_or_buf": str(args.output_file),
        "index": False,
    }

    if args.output_file.exists():
        print(f"Output file {args.output_file} exists. Appending to it.")

        kwargs["mode"] = "a"
        kwargs["header"] = False

        current_df_columns = get_df_columns(args.output_file)

    else:
        current_df_columns = None

        # current_df = pd.read_csv(args.output_file)

    extracted_df = extract_best_runs(args=args,
                                     df=df,
                                     )

    if current_df_columns is not None:
        extracted_df = extracted_df[:, current_df_columns]

    extracted_df.to_csv(**kwargs)


def extract_best_runs(args, df, heuristic_name=None):
    df.drop_duplicates(inplace=True)

    if args.query is not None:
        df.query(args.query, inplace=True)

    if not isinstance(args.sort_column, list):
        args.sort_column = [args.sort_column]

    extracted_df_buffer = []
    for sort_column in args.sort_column:

        sort_by = [sort_column]
        if sort_column == "r_auc":
            sort_by.append("rem_num")
        elif sort_column == "rem_num":
            sort_by.append("r_auc")

        # sort_by.append("")

        # Sort the dataframe
        df.sort_values(by=sort_by,
                       ascending=(not args.sort_descending),
                       inplace=True,
                       )

        extracted_df = df.groupby("network").head(1)

        extracted_df_buffer.append(extracted_df)

    extracted_df = pd.concat(extracted_df_buffer, ignore_index=True)

    if extracted_df["network"].dtype != str:
        extracted_df["network"] = extracted_df["network"].astype(str)

    if heuristic_name is None:
        heuristic_name = "GDM"

        if "_reinserted" in args.file.name:
            heuristic_name += " +R"

    extracted_df["heuristic"] = heuristic_name
    # print("Output DF", extracted_df)

    # Remove duplicates in case of multiple sorting columns
    extracted_df.drop_duplicates(inplace=True)
    return extracted_df


def parse_parameters(parse_string=None):
    parser = ArgumentParser(
        description=""
    )

    parser.add_argument(
        "-f",
        "--file",
        type=Path,
        default=None,
        required=True,
        help="Input DataFrame file location",
    )
    parser.add_argument(
        "-of",
        "--output-file",
        type=Path,
        default=None,
        required=True,
        help="Output DataFrame file location",
    )
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        default=None,
        required=False,
        help="Query the dataframe",
    )
    parser.add_argument(
        "-s",
        "--sort_column",
        type=str,
        default="r_auc",
        required=False,
        nargs="+",
        help="Column used to sort the entries",
    )
    parser.add_argument(
        "-sa",
        "--sort_descending",
        default=False,
        required=False,
        action="store_true",
        help="Descending sorting",
    )
    args, cmdline_args = parser.parse_known_args(parse_string)

    return args


if __name__ == "__main__":
    args = parse_parameters()

    load_and_clean_df(args)

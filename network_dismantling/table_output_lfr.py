from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from network_dismantling.common.df_helpers import df_reader
from network_dismantling.common.humanize_helper import intword
from network_dismantling.machine_learning.pytorch.grid_output import replace_labels
from network_dismantling.table_output import reorder_heuristics
from network_dismantling.table_output_synth import color_dictionary, names_dict, \
    name_regex_lfr

sns.set_theme(context="paper",
              style="whitegrid",
              palette="deep",
              font="sans-serif",
              font_scale=1,
              color_codes=True,
              # rc={'figure.figsize': (15.21, 10.75)},
              # rc={
              #     'figure.figsize': (11.7, 8.27),
              #     'text.usetex': True,
              #     'text.latex.preamble': r'\usepackage{icomma}',
              #     "font.family": "serif",
              #     "font.serif": [],  # use latex default serif font
              #     "font.sans-serif": ["DejaVu Sans"],  # use a specific sans-serif font
              # },
              )


def load_and_display_df(args):
    df = df_reader(args.file)

    display_df(args, df)


def prettify_network_name(x):
    info = None

    x = x["network"]

    for regex in [name_regex_lfr]:

        try:
            info = regex.parse(x)

            if info is not None:
                break

        except:
            continue

    if info is None:
        raise ValueError(f"Could not parse {x}")

    name = "{} ({})".format(
        names_dict[info["type"]],
        intword(info["num_nodes"]),
    )

    return name, info["type"], info["num_nodes"], info["avg_degree"], info["max_degree"], info["mu"], \
        info["t1"], info["t2"], info["min_comm"], info["max_comm"], info["instance_num"]


def display_df(args, df, print=print):
    # # Query the DFs
    if args.query is not None:
        df.query(args.query, inplace=True)

    if args.sort_column == "average_dmg":
        df["average_dmg"] = (1 - df["lcc_size_at_peak"]) / df["slcc_peak_at"]

    # Filter DFs
    columns = ["network", "heuristic", args.sort_column, "static"]
    df = df.loc[:, columns]

    # Sort DF
    df.sort_values(by=[args.sort_column],
                   ascending=(not args.sort_descending),
                   inplace=True,
                   )

    df.reset_index(inplace=True, drop=True)

    df[["name", "type", "num nodes", "avg degree", "max degree", "mu",
        "t1", "t2", "min comm", "max comm", "instance_num"]] = df.apply(prettify_network_name, axis=1,
                                                                        result_type="expand")

    print(df)
    df["color"] = df[["type", "num nodes"]].apply(lambda x: f'{x["type"]}_{x["num nodes"]}', axis=1)
    df["color"] = df["color"].apply(lambda x: color_dictionary.get(x, x))

    df.loc[df["static"] == False, "heuristic"] = df["heuristic"] + " (dynamic)"

    df["heuristic"] = df["heuristic"].apply(lambda x: x.replace("_", " ").strip().title())
    df.replace({"heuristic": replace_labels}, inplace=True)

    df.columns = [x.title() for x in df.columns]

    if args.index == "Network":
        index = ["Name", "Num Nodes", "Avg Degree", "Max Degree", "Mu",
                 "T1", "T2", "Min Comm", "Max Comm", ]
    else:
        index = ["Heuristic"]

    df = df.pivot_table(index=[i.title() for i in index],
                        columns=args.columns.title(),
                        values=args.sort_column.title(),
                        aggfunc=np.mean,
                        )

    print("PIVOTED TABLE", df)

    if args.row_nan:
        df.dropna(axis='index', inplace=True)
    if args.col_nan:
        df.dropna(axis='columns', inplace=True)

    if args.index == "Network":
        df: pd.DataFrame = df.div(df.min(axis='columns',
                                         skipna=True,
                                         ),
                                  axis="index",
                                  )

        df_sum = df.mean(axis=0).rename('Average')

        df = df.reindex(reorder_heuristics(df_sum),
                        axis="columns",
                        )
        df.loc[('Average',) + tuple([''] * (len(df.index.names) - 1)), :] = df_sum

        output_df = df

    else:
        df = df.div(df.min(axis='index'))

        df_sum = df.mean(axis=1).rename('Average')
        # .sum(axis=1).div(df.shape[1])
        df = df.reindex(reorder_heuristics(df_sum),
                        axis="index",
                        )
        output_df = pd.concat([df, df_sum], axis=1)

    output_df *= 100
    output_df = output_df.round(1)

    df_sum *= 100
    df_sum = df_sum.round(1)

    df.clip(upper=3, inplace=True)

    print(output_df)
    print(df_sum)

    for column in df.columns:
        nan_num = np.count_nonzero(np.isnan(df[column]))
        if nan_num:
            print("{} NaN values in column {}".format(nan_num, column))

    if args.output:
        file = args.output / f"barplot_synth_i{args.index}_c{args.columns}_q{args.query}.csv"

        if not file.parent.exists():
            file.parent.mkdir(parents=True)

        output_df.to_csv(str(file), sep=',', index=True, header=True)

        file = file.with_suffix(".tex")
        output_df.to_latex(str(file), index=True, header=True, sparsify=True, float_format="%.2f")

    else:
        plt.show()


FUNCTION_MAP = {
    'display_df': load_and_display_df,
}

if __name__ == "__main__":
    parser = ArgumentParser(
        description=""
    )

    parser.add_argument(
        '--command',
        type=str,
        choices=FUNCTION_MAP.keys(),
        default="display_df",
        required=False
    )

    parser.add_argument(
        "-f",
        "--file",
        type=Path,
        default=None,
        required=True,
        nargs="+",
        help="Input DataFrame file(s) location",
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
        "-p",
        "--plot",
        default=False,
        required=False,
        action="store_true",
        help="Plot the (single) result and the heuristics on the same network",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        required=False,
        help="Output plot location",
    )
    parser.add_argument(
        "-fh",
        "--heuristics_file",
        type=Path,
        default="./out/df/heuristics.SYNTH.csv",
        required=False,
        help="Heuristics output DataFrame file location",
        nargs="*",
    )

    parser.add_argument(
        "-s",
        "--sort_column",
        type=str,
        default="r_auc",
        required=False,
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

    parser.add_argument(
        "-i",
        "--index",
        type=str,
        default="Network",
        required=False,
        help="Column used as X axis",
    )
    parser.add_argument(
        "-c",
        "--columns",
        type=str,
        default="Heuristic",
        required=False,
        help="Column used as Y axis",
    )

    parser.add_argument(
        "-P",
        "--pivot",
        default=False,
        action="store_true",
        help="Transpose axis",
    )

    parser.add_argument(
        "-rn",
        "--row_nan",
        default=False,
        required=False,
        action="store_true",
        help="Drop any row with NaN values",
    )
    parser.add_argument(
        "-cn",
        "--col_nan",
        default=False,
        required=False,
        action="store_true",
        help="Drop any column with NaN values",
    )

    args, cmdline_args = parser.parse_known_args()

    if not args.file.is_absolute():
        args.file = args.file.resolve()

    if args.pivot:
        buff = args.index
        args.index = args.columns
        args.columns = buff

    FUNCTION_MAP[args.command](args)

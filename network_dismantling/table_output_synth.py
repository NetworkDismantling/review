from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from network_dismantling.common.humanize_helper import intword
from parse import compile

from network_dismantling.common.df_helpers import df_reader
from network_dismantling.machine_learning.pytorch.grid_output import replace_labels
from network_dismantling.table_output import reorder_heuristics

# from network_dismantling.machine_learning.pytorch.test_networks_table import intword

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

# name_regex = compile("{type}_n{num_nodes:d}_m{num_edges:d}_{directionality}_{instance_num:d}")
name_regex = compile("{type}_n{num_nodes:d}_{}_{instance_num:d}")
name_regex_lfr = compile(
    "{type}_N{num_nodes:d}_k{avg_degree:f}_maxk{max_degree:d}_mu{mu:f}_t1{t1:f}_t2{t2:f}_minc{min_comm:d}_maxc{max_comm:d}{}_i{instance_num:d}")
# LFRp1_N16384_k6.0_maxk128_mu0.1_t12.2_t21.0_minc128_maxc640_S0_i10

blues = sns.color_palette("Blues", 4)
reds = sns.color_palette("Reds", 4)
greens = sns.color_palette("Greens", 4)
# yellows = sns.color_palette("Yellows", 4)
color_dictionary = {
    'Erdos_Renyi_1000': blues[1],
    'Erdos_Renyi_10000': blues[2],
    'Erdos_Renyi_100000': blues[3],
    'planted_sbm_1000': reds[1],
    'planted_sbm_10000': reds[2],
    'planted_sbm_100000': reds[3],
    'static_power_law_1000': greens[1],
    'static_power_law_10000': greens[2],
    'static_power_law_100000': greens[3],
    # 'LFRp1_1000': yellows[0],
    # 'LFRp1_10000': yellows[1],
    # 'LFRp1_100000': yellows[2],
}

colors = blues[1:] + reds[1:] + greens[1:]

names_dict = {
    "Erdos_Renyi": "ER",
    "planted_sbm": "SBM",
    "static_power_law": "CM",
    "LFRp1": "LFR p1",
}

synth_nets_filter = \
    '(network.str.contains("static_power_law_") and conv_layers=="40,30,20,10," and heads=="1,1,1,1," and fc_layers=="100,100,1,")' + \
    ' or ' + \
    '(network.str.contains("Erdos_Renyi_") and conv_layers=="40,30,20,10," and heads=="5,5,5,5," and fc_layers=="50,30,30,1,")' + \
    ' or ' + \
    '(network.str.contains("planted_sbm_n") and conv_layers=="40,30,20,10," and heads=="5,5,5,5," and fc_layers=="50,30,30,1,")'


def load_and_display_df(args):
    df = df_reader(args.file)

    display_df(args, df)


def prettify_network_name(x):
    info = None

    x = x["network"]

    for regex in [name_regex, name_regex_lfr]:

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

    return name, info["type"], info["num_nodes"], info["instance_num"]


def display_df(args, df, print=print):
    if args.sort_column == "average_dmg":
        df["average_dmg"] = (1 - df["lcc_size_at_peak"]) / df["slcc_peak_at"]

    # # Query the DFs
    if args.query is not None:
        df.query(args.query, inplace=True)

    # Filter DFs
    columns = ["network", args.sort_column, "static"]

    columns = ["network", "heuristic", args.sort_column, "static"]
    df = df.loc[:, columns]

    # Sort DF
    df.sort_values(by=[args.sort_column], ascending=(not args.sort_descending), inplace=True)

    df = df
    # Get groups
    df.reset_index(inplace=True, drop=True)

    df[["network", "type", "num_nodes", "instance_num"]] = df.apply(prettify_network_name, axis=1,
                                                                    result_type="expand")

    print(df)
    df["color"] = df[["type", "num_nodes"]].apply(lambda x: "{}_{}".format(x["type"], x["num_nodes"]), axis=1)
    df["color"] = df["color"].apply(lambda x: color_dictionary.get(x, x))

    df.loc[df["static"] == False, "heuristic"] = df["heuristic"] + " (dynamic)"

    df["heuristic"] = df["heuristic"].apply(lambda x: x.replace("_", " ").strip().title())
    df.replace({"heuristic": replace_labels}, inplace=True)

    df.columns = [x.title() for x in df.columns]

    df = df.pivot_table(index=args.index.title(),
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
        df: pd.DataFrame

        df = df.div(df.min(axis='columns',
                           skipna=True,
                           ),
                    axis="index",
                    )
        df_sum = df.mean(axis=0).rename('Average')

        df = df.reindex(reorder_heuristics(df_sum),
                        axis="columns",
                        )

        print("df post reindex", df)
        output_df = df.append(df_sum)

    else:
        df = df.div(df.min(axis='index'))

        df_sum = df.mean(axis=1).rename('Average')

        df = df.reindex(reorder_heuristics(df_sum),
                        axis="index",
                        )
        output_df = pd.concat([df, df_sum], axis=1)

    # print(df)
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

    if args.pivot:
        buff = args.index
        args.index = args.columns
        args.columns = buff

    FUNCTION_MAP[args.command](args)

#   This file is part of the Network Dismantling review,
#   proposed in the paper "Robustness and resilience of complex networks"
#   by Oriol Artime, Marco Grassia, Manlio De Domenico, James P. Gleeson,
#   Hernán A. Makse, Giuseppe Mangioni, Matjaž Perc and Filippo Radicchi.
#
#   This is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   The project is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with the code.  If not, see <http://www.gnu.org/licenses/>.


import logging
from argparse import ArgumentParser
from ast import literal_eval
from operator import itemgetter
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from network_dismantling import dismantling_methods
from network_dismantling.common.df_helpers import df_reader
from network_dismantling.common.multiprocessing import TqdmLoggingHandler

column_duplicates = [
    "network",
    "heuristic",
    # "removals",
]

replace_labels = {
    "Machine Learning": "GDM",
    "Machine Learning +R": "GDM +R",
    "Machine Learning + R": "GDM +R",
    "Gndr": "GND +R",
    "Gnd": "GND",
    "Ms": "MS",
    "Msr": "MS +R",
    "Corehd": "CoreHD",
    "Egnd": "EGND",
    "Ei S1": r"EI $\sigma_1$",
    "Ei S2": r"EI $\sigma_2$",

    "Collectiveinfluencel1": r"CI $\ell-1$",
    "Collectiveinfluencel2": r"CI $\ell-2$",
    "Collectiveinfluencel3": r"CI $\ell-3$",

    "Gdm": "GDM",
    "Gdm +R": "GDM +R",
    "Coregdm": "CoreGDM",
    "Coregdm +R": "CoreGDM +R",
    "Finder Nd": "FINDER",

    "Pagerank": "PR",
    "Betweenness Centrality": "BC",
    "Degree": "D",

    "Degree (Dynamic)": "AD",
}

# Define run columns to match the runs
run_columns = [
    # "removals",
    "slcc_peak_at",
    "lcc_size_at_peak",
    "slcc_size_at_peak",
    "r_auc",
    # TODO
    "seed",
    "idx",
    "rem_num",
    # "features"
]

rename_networks = {
    "moreno_blogs_blogs": "moreno_blogs",
    "moreno_health_health": "moreno_health",
    "moreno_train_train": "moreno_train",
    "subelj_cora_cora": "subelj_cora",
    "subelj_jdk_jdk": "subelj_jdk",
    "subelj_jung-j_jung-j": "subelj_jung-j",
}

filtered_columns = ["network", "removals", "model_seed", "seed",
                    # "learning_rate", "weight_decay",
                    "negative_slope",
                    "bias", "concat", "removals_num", "dropout"]


def load_and_display_df(args):
    df = df_reader(args.file, include_removals=True)

    display_df(df, args)


def prepare_df(df, args):
    if args.sort_column == "average_dmg":
        df["average_dmg"] = (1 - df["lcc_size_at_peak"]) / df["slcc_peak_at"]

    sort_by = [args.sort_column]
    if "rem_num" in df.columns:
        if args.sort_column == "r_auc":
            sort_by.append("rem_num")
        elif args.sort_column == "rem_num":
            sort_by.append("r_auc")
    else:
        logger.warning("rem_num not in df.columns, not sorting by rem_num")

    # Sort the dataframe
    df.sort_values(by=sort_by,
                   ascending=(not args.sort_descending),
                   inplace=True,
                   )

    df["idx"] = df.index

    # Remove duplicates
    df.drop_duplicates(subset=column_duplicates,
                       keep="first",
                       inplace=True,
                       )

    if args.query is not None:
        df.query(args.query, inplace=True)

    df.loc[:, "lcc_size_at_peak"] *= 100
    df.loc[:, "slcc_size_at_peak"] *= 100


def display_df(df, args):
    prepare_df(df, args)

    print(f"Storing to {args.output}")

    groups = df.groupby("network")
    for network_name, group_df in groups:

        group_df_filtered = group_df.loc[:, [x for x in group_df.columns if
                                             x not in filtered_columns]
                            ]

        group_df_filtered["heuristic"] = group_df_filtered["heuristic"].apply(
            lambda x: dismantling_methods[x].short_name or x
        )
        logger.info("Network {}, showing first {}/{} runs:\n{}\n".format(network_name,
                                                                         min(args.show_first, group_df.shape[0]),
                                                                         group_df.shape[0],
                                                                         group_df_filtered.set_index("idx")
                                                                         )
                    )

        if args.plot:
            sns.set_theme(context=f"{args.context}",
                          style="ticks",
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

            max_num_removals = 0

            groups = group_df.groupby("heuristic", sort=True)

            # Create new figure
            fig, ax = plt.subplots()

            # zindex = len(groups)
            for function_name, heuristic_df in groups:
                dismantling_method = dismantling_methods[function_name]

                heuristic_name = dismantling_method.short_name
                if heuristic_name is None:
                    heuristic_name = function_name

                if heuristic_df.shape[0] > 1:
                    raise RuntimeError(f"There should be only one row per heuristic. "
                                       f"Found {heuristic_df.shape[0]} for {heuristic_name} in network {network_name}"
                                       )

                heuristic_df.reset_index(drop=True, inplace=True)

                infos = heuristic_df.loc[0, :]

                removals = literal_eval(infos.pop("removals"))
                num_removals = len(removals)
                max_num_removals = max(max_num_removals, num_removals)

                color = dismantling_method.plot_color # or color_mapping.get(heuristic_name, None)
                # marker = dismantling_method.plot_marker or marker_mapping[heuristic_name]
                marker = "o" if dismantling_method.includes_reinsertion else "s"

                # TODO Improve this
                x = list(map(itemgetter(0), removals))
                y = list(map(itemgetter(3), removals))

                plt.plot(x, y,
                         # marker=
                         f'-{marker}',
                         markersize=4,
                         linewidth=2,
                         color=color,
                         # zorder=zindex,
                         label=str(heuristic_name)
                         )
                # zindex -= 1

            fig.gca().set_xbound(lower=1)

            fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

            # Add labels
            plt.xlabel('Number of removed nodes')
            plt.ylabel('LCC Size')

            # Rotate xticks
            plt.xticks(rotation=30)

            # Despine the plot
            sns.despine()

            plt.legend(title="Method",
                       bbox_to_anchor=(1.05, 0.5),
                       loc="center left",
                       borderaxespad=0.,
                       frameon=False,
                       )

            if args.output is None:
                # plt.xlim(right=num_removals * (1.10))
                plt.tight_layout()

                plt.show()
            else:
                plt.xlim(right=max_num_removals * (1.05))

                file = args.output / f"{network_name}.pdf"

                if not file.parent.exists():
                    file.parent.mkdir(parents=True)

                plt.savefig(str(file), bbox_inches='tight')

                # figLegend = pylab.figure()
                #
                # file = args.output / "legend.pdf"
                # # produce a legend for the objects in the other figure
                # pylab.figlegend(*ax.get_legend_handles_labels(),
                #                 ncol=len(removals_list),
                #                 loc='upper left',
                #                 )
                #
                # figLegend.savefig(str(file), bbox_inches='tight')

            plt.close('all')


FUNCTION_MAP = {
    'display_df': load_and_display_df,
}

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(TqdmLoggingHandler())

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
        nargs='+',
        help="Output DataFrame file(s) location",
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
        "-sf",
        "--show_first",
        type=int,
        default=15,
        required=False,
        help="Show first N dismantling curves",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        required=True,
        help="Output plot location",
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
        "-P",
        "--plot",
        default=False,
        required=False,
        action="store_true",
        help="Plot the dismantling curves",
    )

    parser.add_argument(
        "-C",
        "--context",
        default="paper",
        required=False,
        help="Scaling of the plot",
        choices=["paper", "talk", "poster"],
    )
    args, cmdline_args = parser.parse_known_args()

    if args.output is not None:
        args.output = args.output.resolve()

        if args.output.exists() and not args.output.is_dir():
            raise RuntimeError(f"Output path {args.output} is not a directory!")

        if not args.output.exists():
            args.output.mkdir(parents=True)

        args.plot = True

    if cmdline_args:
        logger.warning(f"Unknown arguments: {cmdline_args}")

    FUNCTION_MAP[args.command](args)

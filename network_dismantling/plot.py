from argparse import ArgumentParser
from ast import literal_eval
from operator import itemgetter
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from network_dismantling.common.df_helpers import df_reader

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
    "Ei S1": "EI $\sigma_1$",
    "Ei S2": "EI $\sigma_2$",

    "Collectiveinfluencel1": "CI $\ell-1$",
    "Collectiveinfluencel2": "CI $\ell-2$",
    "Collectiveinfluencel3": "CI $\ell-3$",

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

color_mapping = {
    "CoreGDM": "#3080bd",
    "GDM": "#3080bd",
    "GDM +R": "#084488",
    "GDM AUC": "#3080bd",
    "GDM +R AUC": "#084488",
    "GDM #Removals": "#3080bd",
    "GDM +R #Removals": "#084488",
    "GND +R": "#ff7f0e",
    "GND": "#ffbb78",
    "MS +R": "#2ca02c",
    "MS": "#98df8a",
    "BC": "#8c564b",
    "D": "#9467bd",
    "CI $\ell-2$": "#d62728",
    "CI $\ell-3$": "#d62728",
    "PR": "#ff9896",
    "FINDER": "#9467bd",
    "CoreHD": "#ff9896",
}

marker_mapping = {
    "CoreGDM": "o",
    "GDM": "o",
    "GDM +R": "o",
    "GDM AUC": "o",
    "GDM +R AUC": "o",
    "GDM #Removals": "o",
    "GDM +R #Removals": "o",

    "GND +R": "s",
    "GND": "s",

    "MS +R": "v",
    "MS": "v",

    "BC": "D",
    "PR": "+",
    "D": "*",

    "CI $\ell-2$": "X",
    "CI $\ell-3$": "X",

    "FINDER": "H",
    "CoreHD": "^",
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

    display_df(args, df)


def prepare_df(df, args):
    if args.sort_column == "average_dmg":
        df["average_dmg"] = (1 - df["lcc_size_at_peak"]) / df["slcc_peak_at"]

    sort_by = [args.sort_column]
    if args.sort_column == "r_auc":
        sort_by.append("rem_num")
    elif args.sort_column == "rem_num":
        sort_by.append("r_auc")

    # Sort the dataframe
    df.sort_values(by=sort_by,
                   ascending=(not args.sort_descending),
                   inplace=True,
                   )

    df["idx"] = df.index

    df.drop_duplicates(subset=["network",
                               "heuristic",
                               ],
                       keep="first",
                       inplace=True,
                       )

    if args.query is not None:
        df.query(args.query, inplace=True)

    df.loc[:, "lcc_size_at_peak"] *= 100
    df.loc[:, "slcc_size_at_peak"] *= 100


def display_df(df, print=print):
    prepare_df(df, args)

    groups = df.groupby("network")
    for network_name, group_df in groups:
        group_df_filtered = group_df.loc[:, [x for x in group_df.columns if
                                             x not in filtered_columns]
                            ]

        print("Network {}, showing first {}/{} runs:\n{}\n".format(network_name,
                                                                   min(args.show_first, group_df.shape[0]),
                                                                   group_df.shape[0],
                                                                   group_df_filtered.set_index("idx")
                                                                   )
              )

        removals_list = []
        max_num_removals = 0
        for heuristic_name, heuristic_df in group_df.groupby("heuristic"):
            assert heuristic_df.shape[0] == 1, \
                f"There should be only one row per heuristic. Found {heuristic_df.shape[0]} for {heuristic_name} in network {network_name}"

            heuristic_df.reset_index(drop=True, inplace=True)

            infos = heuristic_df.loc[0, :]

            removals = literal_eval(infos.pop("removals"))
            num_removals = len(removals)
            max_num_removals = max(max_num_removals, num_removals)

            removals_list = [(heuristic_name, removals)]

        # Create new figure
        fig, ax = plt.subplots()
        plt.xticks(rotation=45)

        fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        zindex = len(removals_list)
        for heuristic_name, removals in removals_list:
            heuristic_name = str(heuristic_name.strip().title())
            heuristic_name = replace_labels.get(heuristic_name, heuristic_name)

            color = color_mapping[heuristic_name]

            x = list(map(itemgetter(0), removals))
            y = list(map(itemgetter(3), removals))

            plt.plot(x, y, ('-o' if "+R" not in heuristic_name else '-s'),
                     markersize=4,
                     linewidth=2,
                     color=color,
                     zorder=zindex,
                     label=str(heuristic_name)
                     )
            zindex -= 1

            fig.gca().set_xbound(lower=1)

            # Add labels
            plt.xlabel('Number of removed nodes')
            plt.ylabel('LCC Size')

            # Despine the plot
            sns.despine()

            plt.legend(title="Method",
                       bbox_to_anchor=(1.05, 0.5),
                       loc="center left",
                       borderaxespad=0.,
                       frameon=False,
                       )

            plt.tight_layout()

            if args.output is None:
                # plt.xlim(right=num_removals * (1.10))
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
    sns.set_theme(context="talk",
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
        required=False,
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

    args, cmdline_args = parser.parse_known_args()

    FUNCTION_MAP[args.command](args)

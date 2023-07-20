from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from network_dismantling.common.df_helpers import df_reader
from network_dismantling.plot import replace_labels, rename_networks

pd.options.mode.chained_assignment = None

# TODO REMOVE THIS FILTER
review_networks = [
    "arenas-meta",
    "dimacs10-celegansneural",
    "foodweb-baydry",
    "foodweb-baywet",
    "maayan-figeys",
    "maayan-foodweb",
    "maayan-Stelzl",
    "maayan-vidal",
    "moreno_propro",
    "cfinder-google",
    "cit-HepPh",
    "citeseer",
    "com-amazon",
    "com-dblp",
    "dblp-cite",
    "dimacs10-polblogs",
    "econ-wm1",
    "linux",
    "p2p-Gnutella06",
    "p2p-Gnutella31",
    "subelj_jdk_jdk",
    "subelj_jung-j_jung-j",
    "web-EPA",
    "web-NotreDame",
    "web-Stanford",
    "web-webbase-2001",
    "wordnet-words",
    "advogato",
    "com-youtube",
    "corruption",
    "digg-friends",
    "douban",
    "ego-twitter",
    "email-EuAll",
    "hyves",
    "librec-ciaodvd-trust",
    "librec-filmtrust-trust",
    "loc-brightkite",
    "loc-gowalla",
    "moreno_crime_projected",
    "moreno_train_train",
    "munmun_digg_reply_LCC",
    "munmun_twitter_social",
    "opsahl-ucsocial",
    "pajek-erdos",
    "petster-cat-household",
    "petster-catdog-household",
    "petster-hamster",
    "slashdot-threads",
    "slashdot-zoo",
    "soc-Epinions1",
    "twitter_LCC",
    "ARK201012_LCC",
    "dimacs9-COL",
    "dimacs9-NY",
    "eu-powergrid",
    "gridkit-eupowergrid",
    "gridkit-north_america",
    "inf-USAir97",
    "internet-topology",
    "london_transport_multiplex_aggr",
    "opsahl-openflights",
    "opsahl-powergrid",
    "oregon2_010526",
    "power-eris1176",
    "roads-california",
    "roads-northamerica",
    "roads-sanfrancisco",
    "route-views",
    "tech-RL-caida",
]

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


def load_and_display_df(args):
    df = df_reader(args.file,
                   include_removals=False,
                   )

    display_df(args, df)


def display_df(args, df, print=print):
    if args.sort_column == "average_dmg":
        df["average_dmg"] = (1 - df["lcc_size_at_peak"]) / df["slcc_peak_at"]

    if args.query is not None:
        df.query(args.query, inplace=True)

    # Filter DFs
    columns = ["network", "heuristic", args.sort_column, "static"]
    df = df.loc[:, columns]

    # Sort DF
    df.sort_values(by=[args.sort_column],
                   ascending=(not args.sort_descending),
                   inplace=True,
                   )

    # # TODO REMOVE THIS FILTER
    # df = df.loc[(df["network"].isin(review_networks))]

    df.reset_index(inplace=True, drop=True)

    df.loc[df["static"] == False, "heuristic"] = df["heuristic"] + " (dynamic)"

    df["heuristic"] = df["heuristic"].apply(lambda x: x.replace("_", " ").strip().title())
    df["network"] = df["network"].apply(lambda x: rename_networks.get(x, x))

    # Rename heuristic names
    df.replace(
        {
            "heuristic": replace_labels
        },
        inplace=True
    )

    df.columns = [x.title() for x in df.columns]

    df = df.pivot_table(index=args.index.title(),
                        columns=args.columns.title(),
                        values=args.sort_column.title(),
                        )

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

        output_df = df.append(df_sum)

    else:
        df = df.div(df.min(axis='index'))

        df_sum = df.mean(axis=1).rename('Average')

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
        file = args.output / (
            f"barplot_i{args.index}_c{args.columns}_q{args.query}.csv"
        )

        if not file.parent.exists():
            file.parent.mkdir(parents=True)

        output_df.to_csv(str(file), sep=',', index=True, header=True)

        s = df.style.highlight_min(
            # cellcolor:[HTML]{FFFF00};
            # itshape:;
        )
        file = file.with_suffix(".tex")

        s.to_latex(
            str(file),
            # column_format="rrrrr",
            position="h",
            position_float="centering",
            hrules=True,
            # label="table:5",
            # caption="Styled LaTeX Table",
            multirow_align="t",
            multicol_align="r",
        )
        output_df.to_latex(str(file), index=True, header=True, sparsify=True)  # , float_format="%.0f")

        # # create a second figure for the legend
        # figLegend = pylab.figure()
        #
        # file = file.with_suffix(".legend.pdf")
        # # produce a legend for the objects in the other figure
        # pylab.figlegend(*fig.get_legend_handles_labels(), loc='upper left')
        #
        # figLegend.savefig(str(file), bbox_inches='tight')
    else:
        plt.show()


def reorder_heuristics(df_sum):
    # TODO improve this, for instance by using the dismantler_methods information

    sum_values = df_sum.to_dict()
    # order = list(map(itemgetter(0), sorted(sum_values.items(), key=itemgetter(1))))
    order = sorted(sum_values.keys())

    return \
            [x for x in order if (not ("+R" in x or "CoreHD" in x or "CI" in x or "FINDER" in x))] + \
            [x for x in order if ("+R" in x or "CoreHD" in x or "CI" in x or "FINDER" in x)]


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
        nargs="*",
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

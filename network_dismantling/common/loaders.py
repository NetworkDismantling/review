from pathlib import Path
from typing import Union


def load_graph(
    file: Union[Path, str],
    fmt="auto",
    ignore_vp=None,
    ignore_ep=None,
    ignore_gp=None,
    directed=True,
    **kwargs
):
    import warnings
    from graph_tool import load_graph_from_csv

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        from graph_tool import load_graph

    if (
        fmt == "auto"
        and isinstance(file, str)
        and Path(file).suffix[1:] in ["csv", "edgelist", "edge", "edges", "el", "txt"]
    ):
        delimiter = kwargs.get("delimiter", None)
        if delimiter is None:
            delimiter = "," if Path(file).suffix == ".csv" else " "

        g = load_graph_from_csv(
            file,
            directed=directed,
            eprop_types=kwargs.get("eprop_types", None),
            eprop_names=kwargs.get("eprop_names", None),
            # string_vals=kwargs.get("string_vals", False),
            hashed=kwargs.get("hashed", False),
            hash_type=kwargs.get("hash_type", "string"),
            skip_first=kwargs.get("skip_first", False),
            ecols=kwargs.get("ecols", (0, 1)),
            csv_options=kwargs.get(
                "csv_options", {"delimiter": delimiter, "quotechar": '"'}
            ),
        )
    else:
        g = load_graph(
            file,
            fmt=fmt,
            ignore_vp=ignore_vp,
            ignore_ep=ignore_ep,
            ignore_gp=ignore_gp,
            **kwargs
        )

    return g

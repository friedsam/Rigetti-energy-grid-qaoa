import os
import networkx as nx
import pandas as pd


def load_graph_from_parquet(path=None):
    path = path or os.environ.get("DATA_URI") or os.environ.get("DATA_PATH")
    if not path:
        raise ValueError("Set DATA_PATH or DATA_URI.")

    df = pd.read_parquet(path)

    required = {"node_1", "node_2", "weight"}
    if not required.issubset(df.columns):
        raise ValueError(f"Expected columns {required}, got {df.columns}")

    G = nx.Graph()

    for u, v, w in df[["node_1", "node_2", "weight"]].itertuples(index=False):
        G.add_edge(int(u), int(v), weight=float(w))

    return G
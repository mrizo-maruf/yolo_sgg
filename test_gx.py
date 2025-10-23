import itertools as it
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def draw_labeled_multigraph(G, attr_name='label', ax=None):
    """
    Draw a multigraph with labeled edges.
    """
    connectionstyle = [f"arc3,rad={r}" for r in it.accumulate([0.15] * 4)]

    pos = nx.shell_layout(G)
    nx.draw_networkx_nodes(G, pos, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=20, ax=ax)
    nx.draw_networkx_edges(
        G, pos, edge_color="grey", connectionstyle=connectionstyle, ax=ax
    )

    labels = {
        tuple(edge): attrs[attr_name]
        for *edge, attrs in G.edges(keys=True, data=True)
    }
    nx.draw_networkx_edge_labels(
        G,
        pos,
        labels,
        connectionstyle=connectionstyle,
        label_pos=0.3,
        font_color="blue",
        bbox={"alpha": 0},
        ax=ax,
    )


# Create one example graph
nodes = "ABC"
prod = list(it.product(nodes, repeat=2))
pairs = prod * 2  # Create pairs with duplicates

G = nx.MultiDiGraph()
for i, (u, v) in enumerate(pairs):
    G.add_edge(u, v, w=round(i / 3, 2))

# Create single plot
fig, ax = plt.subplots(figsize=(12, 8))
draw_labeled_multigraph(G, "w")
ax.set_title("Multigraph Example")
plt.tight_layout()
plt.show()
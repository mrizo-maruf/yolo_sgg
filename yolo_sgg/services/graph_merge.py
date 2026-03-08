import networkx as nx

import YOLOE.utils as yutils


class GraphMergeService:
    @staticmethod
    def merge(persistent_graph: nx.MultiDiGraph, current_graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
        return yutils.merge_scene_graphs(persistent_graph, current_graph)

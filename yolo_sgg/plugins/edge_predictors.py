from dataclasses import dataclass

from ssg.ssg_main import edges


@dataclass
class SceneVerseEdgePredictor:
    name: str = "sceneverse"

    def predict(self, current_graph, frame_objs, T_w_c, depth_m) -> None:
        edges(current_graph, frame_objs, T_w_c, depth_m)

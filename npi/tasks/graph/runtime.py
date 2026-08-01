from dataclasses import dataclass

from npi.core.runtime import RecursiveExecutor
from npi.tasks.graph.codec import CODEC
from npi.tasks.graph.environment import GraphEnvironment
from npi.tasks.graph.spec import SPEC


@dataclass(frozen=True)
class DijkstraResult:
    distances: list[int | None]
    parents: list[int | None]
    model_steps: int
    maximum_depth: int


def execute_dijkstra(model, node_count, edges, source, *, use_xla=True):
    environment = GraphEnvironment(node_count, list(edges), source)
    executor = RecursiveExecutor(
        model,
        SPEC,
        CODEC,
        environment,
        step_limit=200 * (node_count * node_count + len(edges) + 1),
        use_xla=use_xla,
    )
    stats = executor.execute()
    return DijkstraResult(
        environment.distances(),
        environment.parents(),
        stats.model_steps,
        stats.maximum_depth,
    )

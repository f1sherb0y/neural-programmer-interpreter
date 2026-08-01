from npi.core.runtime import RecursiveExecutor
from npi.tasks.graph_ram.codec import CODEC
from npi.tasks.graph_ram.environment import RamGraphEnvironment
from npi.tasks.graph_ram.spec import SPEC


def execute_dijkstra(model, node_count, edges, source, *, use_xla=True):
    environment = RamGraphEnvironment(node_count, list(edges), source)
    limit = 1_000 * (node_count + len(edges) + 1) * max(1, node_count.bit_length())
    executor = RecursiveExecutor(
        model,
        SPEC,
        CODEC,
        environment,
        step_limit=limit,
        use_xla=use_xla,
    )
    stats = executor.execute()
    distances, parents = environment.result()
    return distances, parents, stats

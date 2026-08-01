from npi.core.spec import Arguments
from npi.core.traces import Decision, Episode
from npi.tasks.graph.environment import NULL, GraphEnvironment
from npi.tasks.graph.spec import (
    DEFAULT_ARGUMENTS,
    Opcode,
    Program,
    Symbol,
)


class DijkstraTrace:
    def __init__(
        self,
        node_count: int,
        weighted_edges: list[tuple[int, int, int]],
        source: int,
    ):
        self.environment = GraphEnvironment(node_count, weighted_edges, source)
        self.episodes: list[Episode] = []
        self._run_dijkstra()

    def _episode(
        self,
        program: Program,
        arguments: Arguments = DEFAULT_ARGUMENTS,
    ) -> Episode:
        episode = Episode(program, tuple(map(int, arguments)))
        self.episodes.append(episode)
        return episode

    def _call(
        self,
        episode: Episode,
        child: Program,
        arguments: Arguments = DEFAULT_ARGUMENTS,
    ) -> None:
        episode.decisions.append(
            Decision(
                observation=self.environment.observe(),
                end=False,
                next_program=child,
                next_arguments=tuple(map(int, arguments)),
                has_child=True,
            )
        )
        dispatch = {
            Program.INITIALIZE: self._run_initialize,
            Program.FIND_MIN: self._run_find_min,
            Program.SETTLE: self._run_settle,
            Program.SCAN_EDGES: self._run_scan_edges,
            Program.RELAX: self._run_relax,
            Program.ACT: lambda: self._run_act(arguments),
        }
        dispatch[child]()

    def _end(self, episode: Episode) -> None:
        episode.decisions.append(
            Decision(
                observation=self.environment.observe(),
                end=True,
                next_program=int(Program.ACT),
                next_arguments=DEFAULT_ARGUMENTS,
                has_child=False,
            )
        )

    def _act(
        self,
        episode: Episode,
        opcode: Opcode,
        first: Symbol,
        second: Symbol,
        third: Symbol = Symbol.DEFAULT,
    ) -> None:
        self._call(
            episode,
            Program.ACT,
            (int(opcode), int(first), int(second), int(third)),
        )

    def _run_dijkstra(self) -> None:
        episode = self._episode(Program.DIJKSTRA)
        self._call(episode, Program.INITIALIZE)
        while True:
            self._call(episode, Program.FIND_MIN)
            if self.environment.pointer_registers[Symbol.BEST] == NULL:
                break
            self._act(
                episode,
                Opcode.READ_VAL,
                Symbol.D_BEST,
                Symbol.BEST,
                Symbol.DISTANCE,
            )
            self._act(
                episode,
                Opcode.CMP_VAL,
                Symbol.D_BEST,
                Symbol.INFINITY,
            )
            if self.environment.comparison[1]:
                break
            self._call(episode, Program.SETTLE)
            self._call(episode, Program.SCAN_EDGES)
        self._end(episode)

    def _run_initialize(self) -> None:
        episode = self._episode(Program.INITIALIZE)
        self._act(
            episode,
            Opcode.WRITE_VAL,
            Symbol.SOURCE,
            Symbol.DISTANCE,
            Symbol.ZERO,
        )
        self._end(episode)

    def _run_find_min(self) -> None:
        episode = self._episode(Program.FIND_MIN)
        self._act(
            episode,
            Opcode.COPY_PTR,
            Symbol.NODE,
            Symbol.FIRST,
        )
        self._act(
            episode,
            Opcode.COPY_PTR,
            Symbol.BEST,
            Symbol.NULL,
        )
        while self.environment.pointer_registers[Symbol.NODE] != NULL:
            node = self.environment._node_for_register(Symbol.NODE)
            if not node.settled:
                if self.environment.pointer_registers[Symbol.BEST] == NULL:
                    self._act(
                        episode,
                        Opcode.COPY_PTR,
                        Symbol.BEST,
                        Symbol.NODE,
                    )
                else:
                    self._act(
                        episode,
                        Opcode.READ_VAL,
                        Symbol.D_V,
                        Symbol.NODE,
                        Symbol.DISTANCE,
                    )
                    self._act(
                        episode,
                        Opcode.READ_VAL,
                        Symbol.D_BEST,
                        Symbol.BEST,
                        Symbol.DISTANCE,
                    )
                    self._act(
                        episode,
                        Opcode.CMP_VAL,
                        Symbol.D_V,
                        Symbol.D_BEST,
                    )
                    if self.environment.comparison[0]:
                        self._act(
                            episode,
                            Opcode.COPY_PTR,
                            Symbol.BEST,
                            Symbol.NODE,
                        )
            self._act(
                episode,
                Opcode.FOLLOW_PTR,
                Symbol.NODE,
                Symbol.NODE,
                Symbol.NEXT_NODE,
            )
        self._end(episode)

    def _run_settle(self) -> None:
        episode = self._episode(Program.SETTLE)
        self._act(
            episode,
            Opcode.COPY_PTR,
            Symbol.U,
            Symbol.BEST,
        )
        self._act(
            episode,
            Opcode.WRITE_BIT,
            Symbol.U,
            Symbol.SETTLED,
            Symbol.BIT_ONE,
        )
        self._act(
            episode,
            Opcode.FOLLOW_PTR,
            Symbol.EDGE,
            Symbol.U,
            Symbol.FIRST_EDGE,
        )
        self._end(episode)

    def _run_scan_edges(self) -> None:
        episode = self._episode(Program.SCAN_EDGES)
        while self.environment.pointer_registers[Symbol.EDGE] != NULL:
            self._act(
                episode,
                Opcode.FOLLOW_PTR,
                Symbol.V,
                Symbol.EDGE,
                Symbol.NEIGHBOR,
            )
            self._call(episode, Program.RELAX)
            self._act(
                episode,
                Opcode.FOLLOW_PTR,
                Symbol.EDGE,
                Symbol.EDGE,
                Symbol.NEXT_EDGE,
            )
        self._end(episode)

    def _run_relax(self) -> None:
        episode = self._episode(Program.RELAX)
        self._act(
            episode,
            Opcode.READ_VAL,
            Symbol.D_U,
            Symbol.U,
            Symbol.DISTANCE,
        )
        self._act(
            episode,
            Opcode.READ_VAL,
            Symbol.WEIGHT,
            Symbol.EDGE,
            Symbol.EDGE_WEIGHT,
        )
        self._act(
            episode,
            Opcode.ADD_VAL,
            Symbol.CANDIDATE,
            Symbol.D_U,
            Symbol.WEIGHT,
        )
        self._act(
            episode,
            Opcode.READ_VAL,
            Symbol.D_V,
            Symbol.V,
            Symbol.DISTANCE,
        )
        self._act(
            episode,
            Opcode.CMP_VAL,
            Symbol.CANDIDATE,
            Symbol.D_V,
        )
        if self.environment.comparison[0]:
            self._act(
                episode,
                Opcode.WRITE_VAL,
                Symbol.V,
                Symbol.DISTANCE,
                Symbol.CANDIDATE,
            )
            self._act(
                episode,
                Opcode.WRITE_PTR,
                Symbol.V,
                Symbol.PARENT,
                Symbol.U,
            )
        self._end(episode)

    def _run_act(self, arguments: Arguments) -> None:
        episode = self._episode(Program.ACT, arguments)
        self._end(episode)
        self.environment.execute(arguments)

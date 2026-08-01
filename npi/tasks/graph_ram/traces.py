from npi.core.traces import Decision, Episode
from npi.tasks.graph_ram.environment import RamGraphEnvironment
from npi.tasks.graph_ram.spec import DEFAULT_ARGUMENTS, Opcode, Program, Register

U = Register.R0
V = Register.R1
EDGE = Register.R2
D_U = Register.R3
D_V = Register.R4
WEIGHT = Register.R5
CANDIDATE = Register.R6
POP_KEY = Register.R8
POS = Register.R9
OTHER = Register.R10
CHILD = Register.R11
CHILD_KEY = Register.R12
CURRENT_KEY = Register.R13
ADDRESS = Register.R14
SIZE = Register.R15
LAST = Register.R16
OTHER_ADDRESS = Register.R17
BASE = Register.R18
TEMP_NODE = Register.R19
OTHER_NODE = Register.R20
TEMP_KEY = Register.R21
OTHER_KEY = Register.R22
TEMP = Register.R23


class RamDijkstraTrace:
    def __init__(self, node_count, weighted_edges, source):
        self.environment = RamGraphEnvironment(node_count, weighted_edges, source)
        self.episodes: list[Episode] = []
        self._run_dijkstra()

    def _episode(self, program, arguments=DEFAULT_ARGUMENTS):
        episode = Episode(int(program), tuple(map(int, arguments)))
        self.episodes.append(episode)
        return episode

    def _call(self, episode, child, arguments=DEFAULT_ARGUMENTS):
        episode.decisions.append(
            Decision(
                observation=self.environment.observe(),
                end=False,
                next_program=int(child),
                next_arguments=tuple(map(int, arguments)),
                has_child=True,
            )
        )
        dispatch = {
            Program.INITIALIZE: self._run_initialize,
            Program.INSERT: self._run_insert,
            Program.BUBBLE_UP: self._run_bubble_up,
            Program.EXTRACT: self._run_extract,
            Program.BUBBLE_DOWN: self._run_bubble_down,
            Program.SCAN_EDGES: self._run_scan_edges,
            Program.RELAX: self._run_relax,
            Program.ACT: lambda: self._run_act(arguments),
        }
        dispatch[Program(child)]()

    def _end(self, episode):
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
        episode,
        opcode,
        first=Register.DEFAULT,
        second=Register.DEFAULT,
        third=Register.DEFAULT,
    ):
        self._call(
            episode,
            Program.ACT,
            (int(opcode), int(first), int(second), int(third)),
        )

    def _heap_address(self, episode, destination, index):
        self._act(episode, Opcode.LOAD, destination, Register.ONE)
        self._act(episode, Opcode.SHL1, TEMP, index)
        self._act(episode, Opcode.ADD, destination, destination, TEMP)
        self._act(episode, Opcode.ADD, destination, destination, Register.ONE)

    def _read_heap_key(self, episode, destination, index):
        self._heap_address(episode, ADDRESS, index)
        self._act(episode, Opcode.ADD, ADDRESS, ADDRESS, Register.ONE)
        self._act(episode, Opcode.LOAD, destination, ADDRESS)

    def _swap_heap(self, episode, left, right):
        self._heap_address(episode, ADDRESS, left)
        self._heap_address(episode, OTHER_ADDRESS, right)
        self._act(episode, Opcode.LOAD, TEMP_NODE, ADDRESS)
        self._act(episode, Opcode.LOAD, OTHER_NODE, OTHER_ADDRESS)
        self._act(episode, Opcode.STORE, ADDRESS, OTHER_NODE)
        self._act(episode, Opcode.STORE, OTHER_ADDRESS, TEMP_NODE)
        self._act(episode, Opcode.ADD, ADDRESS, ADDRESS, Register.ONE)
        self._act(episode, Opcode.ADD, OTHER_ADDRESS, OTHER_ADDRESS, Register.ONE)
        self._act(episode, Opcode.LOAD, TEMP_KEY, ADDRESS)
        self._act(episode, Opcode.LOAD, OTHER_KEY, OTHER_ADDRESS)
        self._act(episode, Opcode.STORE, ADDRESS, OTHER_KEY)
        self._act(episode, Opcode.STORE, OTHER_ADDRESS, TEMP_KEY)

    def _run_dijkstra(self):
        episode = self._episode(Program.DIJKSTRA)
        self._call(episode, Program.INITIALIZE)
        while True:
            self._act(episode, Opcode.LOAD, BASE, Register.ONE)
            self._act(episode, Opcode.LOAD, SIZE, BASE)
            self._act(episode, Opcode.CMP, SIZE, Register.ZERO)
            if self.environment.comparison[1]:
                break
            self._call(episode, Program.EXTRACT)
            self._act(episode, Opcode.ADD, ADDRESS, U, Register.ONE)
            self._act(episode, Opcode.LOAD, D_U, ADDRESS)
            self._act(episode, Opcode.CMP, POP_KEY, D_U)
            if self.environment.comparison[1]:
                self._call(episode, Program.SCAN_EDGES)
        self._end(episode)

    def _run_initialize(self):
        episode = self._episode(Program.INITIALIZE)
        self._act(episode, Opcode.LOAD, U, Register.ZERO)
        self._act(episode, Opcode.ADD, ADDRESS, U, Register.ONE)
        self._act(episode, Opcode.STORE, ADDRESS, Register.ZERO)
        self._act(episode, Opcode.LOAD, BASE, Register.ONE)
        self._act(episode, Opcode.STORE, BASE, Register.ZERO)
        self._act(episode, Opcode.MOV, V, U)
        self._act(episode, Opcode.MOV, CANDIDATE, Register.ZERO)
        self._call(episode, Program.INSERT)
        self._end(episode)

    def _run_insert(self):
        episode = self._episode(Program.INSERT)
        self._act(episode, Opcode.LOAD, BASE, Register.ONE)
        self._act(episode, Opcode.LOAD, SIZE, BASE)
        self._act(episode, Opcode.MOV, POS, SIZE)
        self._heap_address(episode, ADDRESS, POS)
        self._act(episode, Opcode.STORE, ADDRESS, V)
        self._act(episode, Opcode.ADD, ADDRESS, ADDRESS, Register.ONE)
        self._act(episode, Opcode.STORE, ADDRESS, CANDIDATE)
        self._act(episode, Opcode.ADD, SIZE, SIZE, Register.ONE)
        self._act(episode, Opcode.STORE, BASE, SIZE)
        self._call(episode, Program.BUBBLE_UP)
        self._end(episode)

    def _run_bubble_up(self):
        episode = self._episode(Program.BUBBLE_UP)
        while True:
            self._act(episode, Opcode.CMP, POS, Register.ZERO)
            if self.environment.comparison[1]:
                break
            self._act(episode, Opcode.SUB, OTHER, POS, Register.ONE)
            self._act(episode, Opcode.SHR1, OTHER, OTHER)
            self._read_heap_key(episode, TEMP_KEY, POS)
            self._read_heap_key(episode, OTHER_KEY, OTHER)
            self._act(episode, Opcode.CMP, TEMP_KEY, OTHER_KEY)
            if not self.environment.comparison[0]:
                break
            self._swap_heap(episode, POS, OTHER)
            self._act(episode, Opcode.MOV, POS, OTHER)
        self._end(episode)

    def _run_extract(self):
        episode = self._episode(Program.EXTRACT)
        self._act(episode, Opcode.LOAD, BASE, Register.ONE)
        self._act(episode, Opcode.LOAD, SIZE, BASE)
        self._act(episode, Opcode.SUB, LAST, SIZE, Register.ONE)
        self._act(episode, Opcode.ADD, ADDRESS, BASE, Register.ONE)
        self._act(episode, Opcode.LOAD, U, ADDRESS)
        self._act(episode, Opcode.ADD, ADDRESS, ADDRESS, Register.ONE)
        self._act(episode, Opcode.LOAD, POP_KEY, ADDRESS)
        self._act(episode, Opcode.CMP, LAST, Register.ZERO)
        if self.environment.comparison[1]:
            self._act(episode, Opcode.STORE, BASE, LAST)
        else:
            self._heap_address(episode, ADDRESS, LAST)
            self._act(episode, Opcode.LOAD, TEMP_NODE, ADDRESS)
            self._act(episode, Opcode.ADD, ADDRESS, ADDRESS, Register.ONE)
            self._act(episode, Opcode.LOAD, TEMP_KEY, ADDRESS)
            self._act(episode, Opcode.ADD, ADDRESS, BASE, Register.ONE)
            self._act(episode, Opcode.STORE, ADDRESS, TEMP_NODE)
            self._act(episode, Opcode.ADD, ADDRESS, ADDRESS, Register.ONE)
            self._act(episode, Opcode.STORE, ADDRESS, TEMP_KEY)
            self._act(episode, Opcode.STORE, BASE, LAST)
            self._act(episode, Opcode.MOV, SIZE, LAST)
            self._call(episode, Program.BUBBLE_DOWN)
        self._end(episode)

    def _run_bubble_down(self):
        episode = self._episode(Program.BUBBLE_DOWN)
        self._act(episode, Opcode.MOV, POS, Register.ZERO)
        while True:
            self._act(episode, Opcode.SHL1, CHILD, POS)
            self._act(episode, Opcode.ADD, CHILD, CHILD, Register.ONE)
            self._act(episode, Opcode.CMP, CHILD, SIZE)
            if not self.environment.comparison[0]:
                break
            self._act(episode, Opcode.ADD, OTHER, CHILD, Register.ONE)
            self._act(episode, Opcode.CMP, OTHER, SIZE)
            if self.environment.comparison[0]:
                self._read_heap_key(episode, CHILD_KEY, CHILD)
                self._read_heap_key(episode, OTHER_KEY, OTHER)
                self._act(episode, Opcode.CMP, OTHER_KEY, CHILD_KEY)
                if self.environment.comparison[0]:
                    self._act(episode, Opcode.MOV, CHILD, OTHER)
            self._read_heap_key(episode, CURRENT_KEY, POS)
            self._read_heap_key(episode, CHILD_KEY, CHILD)
            self._act(episode, Opcode.CMP, CHILD_KEY, CURRENT_KEY)
            if not self.environment.comparison[0]:
                break
            self._swap_heap(episode, POS, CHILD)
            self._act(episode, Opcode.MOV, POS, CHILD)
        self._end(episode)

    def _run_scan_edges(self):
        episode = self._episode(Program.SCAN_EDGES)
        self._act(episode, Opcode.LOAD, EDGE, U)
        while True:
            self._act(episode, Opcode.CMP, EDGE, Register.NULL)
            if self.environment.comparison[1]:
                break
            self._act(episode, Opcode.ADD, ADDRESS, EDGE, Register.ONE)
            self._act(episode, Opcode.LOAD, V, ADDRESS)
            self._call(episode, Program.RELAX)
            self._act(episode, Opcode.LOAD, EDGE, EDGE)
        self._end(episode)

    def _run_relax(self):
        episode = self._episode(Program.RELAX)
        self._act(episode, Opcode.ADD, ADDRESS, U, Register.ONE)
        self._act(episode, Opcode.LOAD, D_U, ADDRESS)
        self._act(episode, Opcode.ADD, ADDRESS, EDGE, Register.TWO)
        self._act(episode, Opcode.LOAD, WEIGHT, ADDRESS)
        self._act(episode, Opcode.ADD, CANDIDATE, D_U, WEIGHT)
        self._act(episode, Opcode.ADD, ADDRESS, V, Register.ONE)
        self._act(episode, Opcode.LOAD, D_V, ADDRESS)
        self._act(episode, Opcode.CMP, CANDIDATE, D_V)
        if self.environment.comparison[0]:
            self._act(episode, Opcode.STORE, ADDRESS, CANDIDATE)
            self._act(episode, Opcode.ADD, ADDRESS, V, Register.TWO)
            self._act(episode, Opcode.STORE, ADDRESS, U)
            self._call(episode, Program.INSERT)
        self._end(episode)

    def _run_act(self, arguments):
        episode = self._episode(Program.ACT, arguments)
        self._end(episode)
        self.environment.execute(arguments)
